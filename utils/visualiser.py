import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import constants as c
import os
import json

def load_data(image_name):
    nifti_path = os.path.join(c.IMAGE_FOLDER, image_name + '.nii.gz')
    nifti_data, affine = load_nifti(nifti_path)
    with open(os.path.join(c.LABELS_FOLDER, image_name + '.json')) as file:
        landmark_data = json.load(file)
    landmark_data = {k: landmark_data[k] for k in ['R', 'L', 'N'] if k in landmark_data}
    voxels = world_to_voxel(landmark_data, affine=affine)
    return nifti_data, voxels

def world_to_voxel(landmarks, affine):
    inv_affine = np.linalg.inv(affine)
    voxel_landmarks = {}

    for label, coords in landmarks.items():
        flipped_coords = [coords[0], -coords[1], coords[2]]  # Flip Y coordinate
        voxel_coords = np.dot(inv_affine, np.append(flipped_coords, 1))[:3]
        voxel_landmarks[label] = voxel_coords.tolist()

    return voxel_landmarks

def load_nifti(file_path):
    nii_img = nib.load(file_path)
    affine = nii_img.affine  # Get transformation matrix
    return nii_img.get_fdata(), affine

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
        for label, (x, y, z) in landmark_data.items():
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