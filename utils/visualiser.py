import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import scipy
import utils.geometry_fitting as geom
from scipy.ndimage import map_coordinates
from utils.logger import MedicalLogger
from glob import glob
import constants as c
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.geometry_fitting import split_and_approximate_curve, sample_bezier_curve_3d, LeafletGeometry
from bin.DataLoader import _map_landmarks, _world_to_voxel
def _create_slice_viewer(nifti_data, landmark_data, bezier_curves, control_points):
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
        if orientation == 'xz':
            for curve in bezier_curves:
                y_values = [point[1] for point in curve]
                unique_y = set(y_values)
                print(unique_y)
                if len(unique_y) == 1 and list(unique_y)[0] == slice_idx:
                    x_values = [point[0] for point in curve]
                    z_values = [point[2] for point in curve]
                    fig.add_trace(go.Scatter(x=x_values, y=z_values, mode='lines', line=dict(color='cyan'), name='Curve'))
            
            for point in control_points:
                unique_y = point[1]
                if unique_y == slice_idx:
                    (x, y, z) = point
                    fig.add_trace(go.Scatter(
                        x=[x], y=[z],
                        mode='markers+text',
                        marker=dict(color='orange', size=6),
                        text='G_i',
                        textposition='top center'
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
                            mode='markers',
                            marker=dict(color='blue', size=6)
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
            height=1200,  # Optional, you can adjust this as needed
            width=1200    # Optional, ensuring square aspect ratio
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
    nifti_data, affine, landmarks = dataLoader.load_data(image_name=image_name, trim_image=False)
    voxels = _map_landmarks(landmarks, _world_to_voxel, np.linalg.inv(affine))
    point_filter = ['R', 'RLC', 'RNC']

    rotated_image, rotated_voxels = align_surface(nifti_data, voxels, point_filter)

    RCI_left, _, RCI_right, _ = split_and_approximate_curve(rotated_voxels['RCI'], middle_point=np.array(rotated_voxels['R']))

    Bezier_RCI_left = sample_bezier_curve_3d(RCI_left, granularity=100)
    Bezier_RCI_right = sample_bezier_curve_3d(RCI_right, granularity=100)
    # Project RCI onto plane.
    y_level = round(rotated_voxels['R'][1])
    print(y_level)
    curve = np.array(rotated_voxels['RCI'])

    X = curve[:,0]
    Y = curve[:,1]
    Z = curve[:,2]

    new_points = []
    for point in rotated_voxels['RCI']:
        new_point = [point[0], y_level, point[2]]
        new_points.append(new_point)
    rotated_voxels['RCI'] = new_points

    bezier_curves = []
    for curve in [Bezier_RCI_left, Bezier_RCI_right]:
        new_curve = []
        for point in curve:
            new_point = [point[0], y_level, point[2]]
            new_curve.append(new_point)
        bezier_curves.append(new_curve)
    
    control_points = []
    for point in [RCI_left[1], RCI_right[1]]:
        new_point = [point[0], y_level, point[2]]
        control_points.append(new_point)

    app = _create_slice_viewer(rotated_image, rotated_voxels, bezier_curves, control_points)
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

    # Train data
    train_dfs = []
    val_dfs = []
    configs = []
    run_ids = []

    for file in train_files:
        if os.path.isfile(file):
            train_df, val_df, config, run_id = logger.load_from_hdf5(file)

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

    internal_stats = process_metric_data(test_val_dfs, test_configs, test_run_ids, title_suffix=" (Internal)")
    external_stats = process_metric_data(test_ext_val_dfs, test_ext_configs, test_ext_run_ids, title_suffix=" (External)")

    print_metric_summaries(internal_stats, title="Internal Test Metric Differences")
    print_metric_summaries(external_stats, title="External Test Metric Differences")

def print_metric_summaries(summaries, title="Validation Metric Differences"):
    """
    Print metric comparison summaries (mean ± std) for each run.

    Parameters:
    - summaries: list of dicts from process_metric_data()
                 Each should have keys: 'run_id', 'config', 'differences'
    - title: optional overall title for the printout
    """
    print(f"\n=== {title} ===\n")

    for summary in summaries:
        run_label = summary.get("run_id")
        print(f"--- {run_label} ---")
        
        for metric, result in summary["differences"].items():
            print(f"{metric:<30}: {result}")
        
        print()  # newline for spacing between runs

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
    plt.xlim((10,100))
    plt.ylim((2,4))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, plot_name) + ".png", format='png')
    else:
        plt.show()

def boxplot_test_errors(val_dfs, configs, run_ids, title_suffix="", save_path=None, plot_name=None):
    runs = []

     # --- New: Compute baseline naive error ---
    naive_errors = []
    for val_df in val_dfs:
        if 'naive_distance_mm' in val_df.columns:
            val_df = val_df.copy()
            naive_errors.append(val_df['naive_distance_mm'].dropna().values)

    if naive_errors:
        # Stack all and compute global mean
        naive_errors_all = np.concatenate(naive_errors)
        baseline_naive_error = np.mean(naive_errors_all)
    else:
        baseline_naive_error = None

    CPD_errors = []
    for val_df in val_dfs:
        if 'CPD_distance_mm' in val_df.columns:
            val_df = val_df.copy()
            CPD_errors.append(val_df['CPD_distance_mm'].dropna().values)

    if CPD_errors:
        # Stack all and compute global mean
        CPD_errors_all = np.concatenate(CPD_errors)
        baseline_CPD_error = np.mean(CPD_errors_all)
    else:
        baseline_CPD_error = None
    
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
            print(errors.mean(), experiment)
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

    # Add baseline line if available
    if baseline_naive_error is not None:
        plt.axhline(baseline_naive_error, color='red', linestyle='--', label=f'Baseline Naive Error: {baseline_naive_error:.2f} mm')
        
    if baseline_CPD_error is not None:
        plt.axhline(baseline_CPD_error, color='blue', linestyle='--', label=f'Baseline CPD Error: {baseline_CPD_error:.2f} mm')
    
    plt.legend()
    plt.ylim((0,8))
    plt.yticks(np.arange(0,9))
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Distribution of Mean Agent Errors per Test Run{title_suffix}")
    plt.ylabel("Mean Error per Agent (mm)")
    plt.grid(True, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, plot_name) + ".png", format='png')
    else:
        plt.show()

def process_metric_data(val_dfs, configs, run_ids, title_suffix=""):
    metrics_cols = [
        "R_cusp_insertion", "L_cusp_insertion", "N_cusp_insertion",
        "R_symmetry_ratio", "L_symmetry_ratio", "N_symmetry_ratio"
        "R_belly_angle", "L_belly_angle", "N_belly_angle",
        "RL_angle", "LN_angle", "NR_angle"
    ]
    summaries = []
    for val_df, config, run_id in zip(val_dfs, configs, run_ids):
        summary = {}
        ground_truth = {}
        predicted = {}

        for col in metrics_cols:
            true_col = f"true_{col}"
            pred_col = f"pred_{col}"

            if true_col in val_df.columns and pred_col in val_df.columns:
                diff = np.abs(val_df[pred_col] - val_df[true_col])
                mean_diff = diff.mean()
                std_diff = diff.std()
                summary[col] = f"{mean_diff:.2} ± {std_diff:.2f}"
                predicted[col] = f"{val_df[pred_col].mean():.4} ± {val_df[pred_col].std():.2f}"
                ground_truth[col] = f"{val_df[true_col].mean():.4} ± {val_df[true_col].std():.2f}"
            else:
                summary[col] = "N/A"
        summaries.append({
            "run_id": run_id,
            "differences": summary,
            "predicted" : predicted,
            "ground_truth" : ground_truth
        })

    return summaries

def extract_plane_slice_with_points_full(image, p0, p1, p2, points_to_project, size=100):
    """
    Like extract_plane_slice_with_points, but returns u, v, origin for projecting more points.
    """
    image = np.transpose(image, (2, 1, 0))

    v1 = np.array(p1) - np.array(p0)
    v2 = np.array(p2) - np.array(p0)
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    u = v1 / np.linalg.norm(v1)
    v = np.cross(normal, u)

    center = (np.array(p0) + np.array(p1) + np.array(p2)) / 3.0

    grid_lin = np.linspace(-size/2, size/2, size)
    grid_x, grid_y = np.meshgrid(grid_lin, grid_lin)
    grid_voxels = center + grid_x[..., None] * u + grid_y[..., None] * v

    coords = np.stack([
        grid_voxels[..., 2],
        grid_voxels[..., 1],
        grid_voxels[..., 0]
    ], axis=0)

    slice_image = map_coordinates(image, coords, order=1, mode='nearest')

    # Project the main control points
    projected_pts = []
    for pt in points_to_project:
        rel = np.array(pt) - center
        x = np.dot(rel, u)
        y = np.dot(rel, v)
        projected_pts.append((x, y))

    return slice_image, grid_x, grid_y, projected_pts, u, v, center

def visualize_leaflet_planes(image, affine, landmarks, extra_points=None, pred_curves=None, true_curves=None, plot_squares=False):
    from numpy.linalg import inv

    inv_affine = inv(affine)

    def to_voxel(name):
        return _world_to_voxel(landmarks[name], inv_affine)

    def project_point_to_plane(point, origin, u, v):
        rel = np.array(point) - origin
        x = np.dot(rel, u)
        y = np.dot(rel, v)
        return x, y

    leaflets = {
        "Right": ['RLC', 'R', 'RNC'],
        "Left": ['LNC', 'L', 'RLC'],
        "Non-coronary": ['RNC', 'N', 'LNC'],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (ax, (leaflet_name, keys)) in enumerate(zip(axes, leaflets.items())):
        p0 = to_voxel(keys[0])
        p1 = to_voxel(keys[1])
        p2 = to_voxel(keys[2])

        control_voxels = [to_voxel(k) for k in keys]
        slice_img, gx, gy, projected_pts, u, v, origin = extract_plane_slice_with_points_full(
            image, p0, p1, p2, control_voxels, size=100
        )

        ax.imshow(slice_img, cmap='gray', origin='lower',
                  extent=[gx.min(), gx.max(), gy.min(), gy.max()])


        ax.plot(projected_pts[0][0], projected_pts[0][1], 'ro')
        ax.text(projected_pts[0][0] + 2, projected_pts[0][1] + 2, 'Commisure', color='red', fontsize=9)
        ax.plot(projected_pts[1][0], projected_pts[1][1], 'ro')
        ax.text(projected_pts[1][0] + 2, projected_pts[1][1] + 2, 'Cusp', color='red', fontsize=9)
        ax.plot(projected_pts[2][0], projected_pts[2][1], 'ro')
        ax.text(projected_pts[2][0] + 2, projected_pts[2][1] + 2, 'Commisure', color='red', fontsize=9)

        # Add extra points if given
        if extra_points is not None:
            pt1 = extra_points[2 * i]
            pt2 = extra_points[2 * i + 1]

            px1, py1 = project_point_to_plane(pt1, origin, u, v)
            px2, py2 = project_point_to_plane(pt2, origin, u, v)

            ax.plot(px1, py1, 'bo')
            ax.text(px1+2, py1+2, 'G_i', color='blue')
            ax.plot(px2, py2, 'bo')
            ax.text(px2+2, py2+2, 'G_i', color='blue')

        # Plot Bezier curve
        if pred_curves is not None:
            curve1 = pred_curves[2*i]  # Nx3
            curve2 = pred_curves[2*i + 1]
            cx, cy = project_curve_to_plane(np.array(curve1), origin, u, v)

            if plot_squares:
                indexes = np.linspace(0, len(cx), 5).astype(int)
                indexes[-1] -= 1

                x_coords = cx[indexes]
                y_coords = cy[indexes]

                ax.scatter(x_coords, y_coords, color='blue', label='Sample Points')
                side_length = 21
                half_side = side_length / 2

                # Draw squares around each point
                for x, y in zip(x_coords, y_coords):
                    square = patches.Rectangle((x - half_side, y - half_side), side_length, side_length,
                                               linewidth=1, edgecolor='blue', facecolor='none')
                    ax.add_patch(square)
            ax.plot(cx, cy, 'r-', linewidth=2, label='Predicted Curves')


            cx, cy = project_curve_to_plane(np.array(curve2), origin, u, v)
            if plot_squares:
                x_coords = cx[indexes]
                y_coords = cy[indexes]

                ax.scatter(x_coords, y_coords, color='blue', label='Sample Points')

                # Draw squares around each point
                for x, y in zip(x_coords, y_coords):
                    square = patches.Rectangle((x - half_side, y - half_side), side_length, side_length,
                                               linewidth=1, edgecolor='blue', facecolor='none')
                    ax.add_patch(square)
            ax.plot(cx, cy, 'r-', linewidth=2)

        if true_curves is not None:
            curve1 = true_curves[2*i]  # Nx3
            curve2 = true_curves[2*i + 1]
            cx, cy = project_curve_to_plane(np.array(curve1), origin, u, v)
            ax.plot(cx, cy, 'g-', linewidth=2, label='True Curves')

            cx, cy = project_curve_to_plane(np.array(curve2), origin, u, v)
            ax.plot(cx, cy, 'g-', linewidth=2)

        ax.set_title(f"{leaflet_name} Leaflet")
        ax.legend()

    plt.tight_layout()
    plt.show()

def project_point_to_plane(point, origin, u, v):
    """
    Project a single 3D point into 2D coordinates of the plane.
    """
    rel = np.array(point) - origin
    x = np.dot(rel, u)
    y = np.dot(rel, v)
    return x, y

def project_curve_to_plane(curve_points, origin, u, v):
    """
    Projects a 3D curve (Nx3) into the 2D plane defined by origin, u, v.
    """
    rel = curve_points - origin
    x = np.dot(rel, u)
    y = np.dot(rel, v)
    return x, y