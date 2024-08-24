"""Create data management tab for the experiment."""

from dash import dcc
from dash import dash_table
from dash import html
import dash_bootstrap_components as dbc



import numpy as np
import torch as th
import io
import base64
from matplotlib import pyplot as plt
from PIL import Image


def tensor_to_image(tensor):
    # Convert tensor to numpy array and transpose to (H, W, C) format
    np_img = tensor.numpy().transpose(1, 2, 0)
    
    # Ensure the values are in the range [0, 255]
    np_img = (np_img * 255).astype(np.uint8)
    
    # Create a figure and plot the tensor as an image
    img = Image.fromarray(np_img)
    buf = io.BytesIO()
    img.save(buf, format='png')
    buf.seek(0)

    # Encode the image to base64
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    print("image_str: ", img_str[:32])
    return img_str



def get_data_samples_management_tab(exp: "Experiment"):
    num_samples = len(exp.train_loader.dataset)
    columns = [
        "label", "predicted_class", "prediction_loss", "prediction_age",
        "exposure_amount", "deny_listed"
    ]
    return html.Div([
        html.Div(id='tabs-content-dataset', children=[
            html.P(f'Data samples management #{num_samples}',
                id='data-management-header'),
            dbc.Alert(
                "Query error occurred. Please check the syntax!",
                id="alert-query-error",
                color="danger",
                dismissable=True,
                fade=True,
                is_open=False,
            ),
            dcc.Input(
                id='ad-hoc-query-input',
                type='text',
                placeholder= \
                    'def denylist_fn(sample_id, age, loss, times, denied, ' + \
                    'prediction, label): return loss <= value_threhsold',
                style={
                    'width': '85vw',
                    'height': '64px',
                    "margin": "5px",
                }
            ),
            dbc.Button(
                    "Run query",
                    id='run-data-query',
                    color='primary', n_clicks=0,
                    style={"width": '24hv'}),
            dash_table.DataTable(
                id='train-tbl',
                data=exp.get_train_records(),
                columns=[{"name": i, "id": i} for i in columns],
                filter_action="native",
                sort_action="native",
                page_action="native",
                row_selectable='single',
                virtualization=True,
                page_size=500,
                style_table={
                    'height': '50vh',
                    'width': '99vw',
                    "margin": "5px",
                    "display": "flex",
                    "align": "center",
                    'padding': '5px',
                },
                style_cell={
                    'textAlign': 'left',
                    'minWidth': '10vw',
                    'maxWidth': '20vw'}
            ),
            html.Div(
                id='image-container',
                children=html.Img(
                    src=f'data:image/png;base64,{tensor_to_image(th.rand(3, 28, 28))}',
                    style={'width': '80%', 'height': 'auto'}
                    )
            ),
        ]),
    ])


def get_checkpoint_loading_context_menu():
    return html.Div(
        id='plots-context-menu',
        children=[
            html.Ul([
                html.Li('Load Checkpoint', id='load-checkpoint'),
            ])
        ],
        style={'display': 'none'}
    )