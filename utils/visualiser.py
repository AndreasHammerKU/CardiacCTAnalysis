import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import scipy

def ras_to_lps(dict):
    converted_dict = {}
    for key, value in dict.items():
        if isinstance(value[0], (list, tuple, np.ndarray)):  # Check if it's a list of points (curve)
            converted_dict[key] = [np.array([-point[0], -point[1], point[2]]).tolist() for point in value]
        else:  # Single point case
            converted_dict[key] = np.array([-value[0], -value[1], value[2]]).tolist()
    return converted_dict
    

def world_to_voxel(landmarks, affine, orientation=('L', 'P', 'S')):
    inv_affine = np.linalg.inv(affine)
    voxel_landmarks = {}

    for label, value in landmarks.items():
        if isinstance(value[0], (list, tuple, np.ndarray)):  # Check if it's a list of points (curve)
            voxel_landmarks[label] = [np.dot(inv_affine, np.append(point, 1))[:3].tolist() for point in value]
        else:  # Single point case
            voxel_landmarks[label] = np.dot(inv_affine, np.append(value, 1))[:3].tolist()
    return voxel_landmarks

def create_slice_viewer(nifti_data, landmark_data):
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
            height=1000,  # Optional, you can adjust this as needed
            width=1000    # Optional, ensuring square aspect ratio
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