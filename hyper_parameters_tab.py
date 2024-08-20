"""Logic that generates the hyper-parameters tab of the UI."""

from dash import dcc
from dash import html
import dash_bootstrap_components as dbc

import base64
import functools



def get_label_and_input_row(
        variable_label: str,
        variable_name: str,
        variable_type: str,
        variable_default: str):
    row = dbc.Row([
        dbc.Col([
            html.H6(
                variable_label,
                style={
                    "color": "black",
                    "height": "auto",
                    "width": "10vw"
                }),
            dcc.Input(
                id=variable_name, type=variable_type, value=variable_default,
                style={
                    "color": "black",
                    "backgroundColor": "#DDD",
                    "height": "auto",
                    "width": "9vw",
                    'borderRadius': '5px',
                }),
            ],
            style={
                "display": 'flex',
                "flexWrap": 'wrap',
                "width": "9vw"
            }
        ),
    ])
    return dbc.Col([row])


def get_pause_play_button():
    with open('./assets/icons/play.png', 'rb') as f:
        encoded_icon_play = base64.b64encode(f.read()).decode('utf-8')
    img_src = 'data:image/png;base64,{}'.format(encoded_icon_play)
    button = dbc.Button(
        id='resume-pause-train-btn',
        color='transparent',
        n_clicks=0,
        children=[
            "Resume Training",
            html.Img(
                src=img_src,
                style={'width': '4vw', 'height': '6vh'}),
        ],
        style={"color": "black", "borderColor": "transparent", }
    )
    return dbc.Col([button])


def get_hyper_params_div(exp: "Experiment") -> html.Div:
    vars_label_names_types_and_defaults = [
        ("Experiment name",  "experiment_name", "text", exp.name),
        ("Training steps to do", "training_steps", "number",
            exp.training_steps_to_do),
        ("Learning rate", "learning_rate", "number", exp.learning_rate),
        ("Batch size", "batch_size", "number", exp.batch_size),
        ("Eval frequency", "eval_freq", "number",
            exp.eval_full_to_train_steps_ratio),
        ("Checkpoint frequency", "ckpt_freq", "number",
            exp.experiment_dump_to_train_steps_ratio),
    ]

    children = []
    for (l, n, t, d) in vars_label_names_types_and_defaults[:3]:
        children.append(get_label_and_input_row(l, n, t, d))
    children.append(get_pause_play_button())
    for (l, n, t, d) in vars_label_names_types_and_defaults[3:]:
        children.append(get_label_and_input_row(l, n, t, d))

    section = html.Div(
        id="hyper-parameters-panel",
        children=[
            dbc.Row(
                id="hyper-params-row",
                children=children,
                style={
                    "textWeight": "bold",
                    "width": "80vw",
                    "display": "flex",
                    "align": "center",
                    'margin': '0 auto',
                    'padding': '5px',
                }
            ),
        ],
        style={
            "backgroundColor": "#DDD",
        }
    )
    return section

@functools.lru_cache
def get_play_button_html_elements(): 
    with open('./assets/icons/play.png', 'rb') as f:
        encoded_icon_play = base64.b64encode(f.read()).decode('utf-8')
    children = (
        "Resume Training",
        html.Img(
            src='data:image/png;base64,{}'.format(encoded_icon_play),
            style={
                'width': '4vw',
                'height': '6vh'
            }
        )
    )
    return children


@functools.lru_cache
def get_pause_button_html_elements():
    with open('./assets/icons/pause.png', 'rb') as f:
        encoded_icon_pause = base64.b64encode(f.read()).decode('utf-8')
    children = (
        "Pause Training",
        html.Img(
            src='data:image/png;base64,{}'.format(encoded_icon_pause),
            style={
                'width': '4vw',
                'height': '6vh'
            }
        )
    )
    return children