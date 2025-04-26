import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import scipy
import utils.geometry_fitting as geom
from utils.logger import MedicalLogger
from glob import glob
import constants as c
import os
import matplotlib.pyplot as plt

def _create_slice_viewer(nifti_data, landmark_data):
    slices = nifti_data.shape[2]
    
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.Div([
            # Dropdown for selecting orientation
            dcc.Dropdown(
                id='orientation-dropdown',
                options=[
                    {'label': 'XY Plane', 'value': 'xy'},
                    {'label': 'YZ Plane', 'value': 'yz'},
                    {'label': 'XZ Plane', 'value': 'xz'}
                ],
                value='xy',  # Default orientation
                style={'width': '50%'}
            )
        ], style={'padding': '10px'}),

        dcc.Slider(
            id='slice-slider',
            min=0,
            max=slices - 1,
            value=slices // 2,
            marks={i: str(i) for i in range(0, slices, max(1, slices // 10))},
            step=1
        ),
        dcc.Graph(id='slice-view')
    ])
    
    @app.callback(
        Output('slice-view', 'figure'),
        Input('slice-slider', 'value'),
        Input('orientation-dropdown', 'value')
    )
    def update_figure(slice_idx, orientation):
        if orientation == 'xy':
            slice_data = nifti_data[:, :, slice_idx]
            title = f'Slice {slice_idx} (XY Plane)'
        elif orientation == 'yz':
            slice_data = nifti_data[slice_idx, :, :]
            title = f'Slice {slice_idx} (YZ Plane)'
        elif orientation == 'xz':
            slice_data = nifti_data[:, slice_idx, :]
            title = f'Slice {slice_idx} (XZ Plane)'

        # Transpose for correct orientation (optional, depends on your NIfTI orientation)
        slice_data = slice_data.T if orientation != 'yz' else slice_data
        
        fig = go.Figure(data=go.Heatmap(
            z=slice_data,
            colorscale='gray'
        ))

        # Add landmark points if they are in the current slice
        for label, value in landmark_data.items():
            if isinstance(value[0], (list, tuple, np.ndarray)):
                for point in value:
                    (x, y, z) = point
                    if (orientation == 'xy' and round(z) == slice_idx) or \
                    (orientation == 'yz' and round(x) == slice_idx) or \
                    (orientation == 'xz' and round(y) == slice_idx):

                        lx, ly = (x, y) if orientation == 'xy' else (y, z) if orientation == 'yz' else (x, z)

                        fig.add_trace(go.Scatter(
                            x=[lx], y=[ly],
                            mode='markers+text',
                            marker=dict(color='blue', size=6),
                            text=label,
                            textposition='top center'
                        ))
            else:
                (x, y, z) = value
                if (orientation == 'xy' and round(z) == slice_idx) or \
                (orientation == 'yz' and round(x) == slice_idx) or \
                (orientation == 'xz' and round(y) == slice_idx):
                    
                    lx, ly = (x, y) if orientation == 'xy' else (y, z) if orientation == 'yz' else (x, z)
                    
                    fig.add_trace(go.Scatter(
                        x=[lx], y=[ly],
                        mode='markers+text',
                        marker=dict(color='red', size=8),
                        text=label,
                        textposition='top center'
                    ))

        # Setting layout options to ensure square aspect ratio
        fig.update_layout(
            title=title, 
            xaxis_title='X', 
            yaxis_title='Y',
            xaxis=dict(scaleanchor="y"),  # This ensures that x and y axes have the same scale
            yaxis=dict(constrain='domain'),
            height=800,  # Optional, you can adjust this as needed
            width=800    # Optional, ensuring square aspect ratio
        )
        
        return fig
    
    return app

def rotate_image(image, R, pivot):
    """Apply rotation to an image"""
    # Convert 3x3 rotation to 4x4 affine transform for 3D images
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = R

    offset = np.array(pivot) - (R @ np.array(pivot))
    rotated_image = scipy.ndimage.affine_transform(image, R, offset=offset, order=3)
    return rotated_image

def rotate_landmarks(voxels_dict, R, points_filter):
    rotated_dict = {}
    pivot = np.array(voxels_dict[points_filter[0]])

    for key, value in voxels_dict.items():
        if isinstance(value[0], (list, tuple, np.ndarray)):  # Check if it's a list of points (curve)
            rotated_dict[key] = [(np.dot(R, (point - pivot)) + pivot).tolist() for point in value]
        else:  # Single point case
            rotated_dict[key] = (np.dot(R, (value - pivot)) + pivot).tolist()

    return rotated_dict
    

def get_roation_matrix(voxel_dict, points_filter):
    assert len(points_filter) == 3

    P1 = np.array(voxel_dict[points_filter[0]])
    P2 = np.array(voxel_dict[points_filter[1]])
    P3 = np.array(voxel_dict[points_filter[2]])


    v1 = P2 - P1
    v2 = P3 - P1
    source = (np.cross(v1, v2)) / np.linalg.norm(np.cross(v1, v2))

    A,B,C = source

    D = -np.dot(source, P1)

    print("Source: ", source)
    target = np.round(source, decimals=0)

    # Compute rotation axis (cross product)
    axis = np.cross(source, target)

    if np.linalg.norm(axis) < 1e-6:
        return np.eye(3)  # No rotation needed

    axis = axis / np.linalg.norm(axis)  # Normalize axis
    angle = np.arccos(np.clip(np.dot(source, target), -1.0, 1.0))  # Compute angle

    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    return R

def nifti_w2v(image, affine):
    inv_affine = np.linalg.inv(affine)
    return inv_affine @ image

def align_surface(image, voxels, point_filter):
    R = get_roation_matrix(voxels, points_filter=point_filter)

    image = rotate_image(image, R, voxels[point_filter[0]])
    voxels = rotate_landmarks(voxels, R, points_filter=point_filter)

    return image, voxels


def create_slice_app(image_name, dataLoader):
    nifti_data, _, voxels = dataLoader.load_data(image_name=image_name, trim_image=False)

    point_filter = ['R', 'RLC', 'RNC']

    rotated_image, rotated_voxels = align_surface(nifti_data, voxels, point_filter)

    # Project RCI onto plane.
    y_level = rotated_voxels['R'][1]

    curve = np.array(rotated_voxels['RCI'])

    X = curve[:,0]
    Y = curve[:,1]
    Z = curve[:,2]

    new_points = []
    for point in rotated_voxels['RCI']:
        new_point = [point[0], y_level, point[2]]
        new_points.append(new_point)
    rotated_voxels['RCI'] = new_points

    app = _create_slice_viewer(rotated_image, rotated_voxels)
    app.run_server(debug=True)

def view_curve(image_name, dataLoader):
    image, affine, voxel_landmarks = dataLoader.load_data(image_name=image_name, trim_image=False)

    geometry = geom.LeafletGeometry(voxel_landmarks)
    geometry.calculate_bezier_curves()

    print("Average error in transformed image {} is {}".format(image_name, geometry.get_average_mse()))
    geometry.plot(plot_label_points=True, plot_control_points=True)
    geometry = geom.LeafletGeometry(landmarks=voxel_landmarks)
    geometry.calculate_bezier_curves()

    print("Average error in rps image {} is {}".format(image_name, geometry.get_average_mse()))

    geometry.plot(plot_label_points=True, plot_control_points=True)


def visualize_from_logs(logger, save_path=None, experiment=c.DQN_LOGS, viz_name=""):


    train_files = glob(os.path.join(c.LOGS_FOLDER, experiment, c.TRAIN_LOGS, '*'))
    test_files = glob(os.path.join(c.LOGS_FOLDER, experiment, c.TEST_LOGS, '*'))
    test_external_files = glob(os.path.join(c.LOGS_FOLDER, experiment, c.TEST_EXTERNAL_LOGS, '*'))
    print(train_files)
    # Train data
    train_dfs = []
    val_dfs = []
    configs = []
    run_ids = []

    for file in train_files:
        if os.path.isfile(file):
            train_df, val_df, config, run_id = logger.load_from_hdf5(file)
            print(train_df.head(5))
            train_dfs.append(train_df)
            val_dfs.append(val_df)
            configs.append(config)
            run_ids.append(run_id)

    # Plot training validation loss
    plot_validation_loss(val_dfs, configs, run_ids, save_path=save_path, plot_name=f"validation-{viz_name}")

    # Test (internal)
    test_val_dfs = []
    test_configs = []
    test_run_ids = []

    for file in test_files:
        if os.path.isfile(file):
            _, val_df, config, run_id = logger.load_from_hdf5(file)
            test_val_dfs.append(val_df)
            test_configs.append(config)
            test_run_ids.append(run_id)

    # Test (external)
    test_ext_val_dfs = []
    test_ext_configs = []
    test_ext_run_ids = []

    for file in test_external_files:
        if os.path.isfile(file):
            _, val_df, config, run_id = logger.load_from_hdf5(file)
            test_ext_val_dfs.append(val_df)
            test_ext_configs.append(config)
            test_ext_run_ids.append(run_id)

    # Plot boxplots
    boxplot_test_errors(test_val_dfs, test_configs, test_run_ids, title_suffix=" (Internal)", save_path=save_path, plot_name=f"internal-test-{viz_name}")
    boxplot_test_errors(test_ext_val_dfs, test_ext_configs, test_ext_run_ids, title_suffix=" (External)", save_path=save_path, plot_name=f"external-test-{viz_name}")

def plot_validation_loss(val_dfs, configs, run_ids, save_path=None, plot_name=None):
    plt.figure(figsize=(12, 6))
    
    for val_df, config, run_id in zip(val_dfs, configs, run_ids):
        if 'train_episode' in val_df.columns and 'avg_err_in_mm' in val_df.columns:
            # Step 1: Compute mean of each list in avg_err_in_mm
            val_df = val_df.copy()
            val_df['mean_agent_error'] = val_df['avg_err_in_mm'].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else np.nan)

            # Step 2: Group by train_episode and average those means
            grouped = val_df.groupby('train_episode')['mean_agent_error'].mean()

            model = config.get("model_type", "UnknownModel")
            experiment = getattr(config.get("experiment", "UnknownExperiment"), "name", str(config.get("experiment")))
            label = f"{model}-{experiment}"

            # Step 3: Plot
            plt.plot(grouped.index, grouped.values, label=label)

    plt.title("Validation Loss Across Runs (Mean of Avg Errors per Agent per Episode)")
    plt.xlabel("Train Episode")
    plt.ylabel("Mean Validation Error (mm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, plot_name) + ".png", format='png')
    else:
        plt.show()

def boxplot_test_errors(val_dfs, configs, run_ids, title_suffix="", save_path=None, plot_name=None):
    runs = []

    for val_df, config, run_id in zip(val_dfs, configs, run_ids):
        if 'avg_err_in_mm' in val_df.columns:
            val_df = val_df.copy()
            val_df['mean_agent_error'] = val_df['avg_err_in_mm'].apply(
                lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) else np.nan
            )

            errors = val_df['mean_agent_error'].dropna().values

            model = config.get("model_type", "UnknownModel")
            experiment_raw = config.get("experiment", "UnknownExperiment")
            experiment = getattr(experiment_raw, "name", str(experiment_raw))

            runs.append({
                "errors": errors,
                "label": f"{experiment}-{model}",
                "experiment": experiment,
                "model_type": model
            })

    # Sort by experiment, then model_type
    sort_order = ["WORK_ALONE", "SHARE_POSITIONS", "SHARE_PAIRWISE", "Network3D", "CommNet"]
    runs.sort(key=lambda r: (sort_order.index(r["experiment"]), sort_order.index(r["model_type"])))

    # Unpack sorted data
    data = [r["errors"] for r in runs]
    labels = [r["label"] for r in runs]

    # Plot boxplot
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels, showfliers=True)

    plt.ylim((0,12))
    plt.yticks(np.arange(0,13))
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Distribution of Mean Agent Errors per Test Run{title_suffix}")
    plt.ylabel("Mean Error per Agent (mm)")
    plt.grid(True, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, plot_name) + ".png", format='png')
    else:
        plt.show()