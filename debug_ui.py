import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import torch
import numpy as np
import io
import base64
from PIL import Image

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Sample data for the table - list of image IDs
data = {
    'ID': [1, 2, 3],
    'Image Name': ['Sample Image 1', 'Sample Image 2', 'Sample Image 3']
}

df = pd.DataFrame(data)

# Generate sample images as tensors
def generate_sample_tensor():
    return torch.rand(3, 28, 28)  # RGB tensor with 3 channels

sample_images = [generate_sample_tensor() for _ in range(3)]

# Function to convert tensor to a Base64-encoded PNG image
def tensor_to_image(tensor):
    # Convert tensor to numpy array and transpose to (H, W, C) format
    np_img = tensor.numpy().transpose(1, 2, 0)
    
    # Ensure the values are in the range [0, 255]
    np_img = (np_img * 255).astype(np.uint8)
    
    # Create a PIL image from the numpy array
    img = Image.fromarray(np_img)
    buf = io.BytesIO()
    img.save(buf, format='png')
    buf.seek(0)

    # Encode the image to base64
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_str

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            dash_table.DataTable(
                id='image-table',
                columns=[{'name': col, 'id': col} for col in df.columns],
                data=df.to_dict('records'),
                row_selectable='single',
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_as_list_view=True,
            ),
            width=6
        ),
        dbc.Col(
            html.Div(id='image-container', children="Select a row to display an image."),
            width=6,
            style={'textAlign': 'center'}
        )
    ])
], fluid=True)

# Callback to update the image based on the selected row
@app.callback(
    Output('image-container', 'children'),
    Input('image-table', 'selected_rows'),
)
def display_image(selected_rows):
    if not selected_rows:
        return "Select a row to display an image."
    
    # Get the selected row index
    selected_row_index = selected_rows[0]
    
    # Get the corresponding tensor image
    img_tensor = sample_images[selected_row_index]
    
    # Convert tensor to image
    encoded_image = tensor_to_image(img_tensor)
    
    return html.Img(src=f'data:image/png;base64,{encoded_image}', style={'width': '80%', 'height': 'auto'})

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
