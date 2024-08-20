import dash
import logging
from dash import dcc, html, Input, Output, State, MATCH, no_update
from dash import ClientsideFunction, dash_table
from flask.logging import create_logger
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from threading import Thread
from dash.exceptions import PreventUpdate
from collections import defaultdict

import re
import numpy as np

import random as r
import ctypes
import threading
import time
import base64

from hyper_parameters_tab import get_hyper_params_div
from interruptible_thread import InterruptibleThread

from cifar_exp import exp


_logger = logging.getLogger("WeightsLabUI")
_logger.setLevel(logging.DEBUG)


# Layout and app initialization
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

with open('./assets/icons/play.png', 'rb') as f:
    encoded_icon_play = base64.b64encode(f.read()).decode('utf-8')
_BUTTON_PLAY_CHILDREN = (
    "Resume Training",
    html.Img(
        src='data:image/png;base64,{}'.format(encoded_icon_play),
        style={
            'width': '4vw',
            'height': '6vh'
        }
    )
)
with open('./assets/icons/pause.png', 'rb') as f:
    encoded_icon_pause = base64.b64encode(f.read()).decode('utf-8')
_BUTTON_PAUSE_CHILDREN = (
    "Pause Training",
    html.Img(
        src='data:image/png;base64,{}'.format(encoded_icon_pause),
        style={
            'width': '4vw',
            'height': '6vh'
        }
    )
)


# thread = InterruptibleThread(
#     unit_work_fn=lambda: exp.train_step_or_eval_full()
# )



def start_training_thread():
    while True:
        print("[thread] loop ...")
        if exp.get_is_training():
            exp.train_step_or_eval_full()
        else:
            exp.pause_event.wait()


training_thread = Thread(target=start_training_thread)
training_thread.start()


app.layout = html.Div([
    html.H1("GrayBox: Interactive experimentation",
            style={'textAlign': 'center'}),
    get_hyper_params_div(exp),
    dcc.Interval(id='heartbeat', interval=5*1000, n_intervals=0),
    dcc.Interval(id='refreshes', interval=2500, n_intervals=0),
])


@app.callback(
    Output('resume-pause-train-btn', 'children', allow_duplicate=True),
    Input('resume-pause-train-btn', 'n_clicks'),
    prevent_initial_call=True
)
def resume_or_pause_train(start_clicks):
    new_children = None
    print("resume_or_pause_train: ", exp)
    if exp.get_is_training():
        new_children = _BUTTON_PLAY_CHILDREN
        exp.pause_event.clear()
    else:
        new_children = _BUTTON_PAUSE_CHILDREN
        exp.pause_event.set()
    exp.toggle_training_status()
    return new_children


@app.callback(
    Output('resume-pause-train-btn', 'children', allow_duplicate=True),
    Input("heartbeat", "n_intervals"),
    prevent_initial_call=True,
)
def update_play_pause_button_icon(n_intervals):
    new_children = None
    print("update_play_pause_btn: ", n_intervals, exp)
    if exp.get_is_training():
        new_children = _BUTTON_PAUSE_CHILDREN
    else:
        new_children = _BUTTON_PLAY_CHILDREN
    return new_children


@app.callback(
    Input("experiment_name", "value"),
)
def update_experiment_name(name):
    print(f"[UI] Update experiment name: {name}")
    # with exp.lock:
    exp.name = name
    return no_update


@app.callback(
    Input("training_steps", "value"),
)
def update_training_steps(steps):
    print(f"[UI] Update training steps: {steps}")
    # with exp.lock:
    exp.set_training_steps_to_do(steps)
    return no_update


if __name__ == '__main__':
    _logger.info("Starting server...")
    app.run_server(debug=True, port=8051)
    _logger.info("Stoping server...")
    
