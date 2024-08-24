import dash
import logging
from dash import dcc, html, Input, Output, State, MATCH, ALL, no_update
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

import torch as th

from prettyfy import hrf
from graybox.modules_with_ops import Conv2dWithNeuronOps
# from graybox.modules_with_ops import LinearWithNeuronOps
# from graybox.tracking import TrackingMode
from hyper_parameters_tab import get_hyper_params_div
from hyper_parameters_tab import get_play_button_html_elements
from hyper_parameters_tab import get_pause_button_html_elements
from model_architecture_tab import get_model_architecture_tab
from model_architecture_tab import get_layer_representation
from model_architecture_tab import get_neuron_context_menu
from model_architecture_tab import get_layer_context_menu
from plots_div import get_plots_div
from data_management_tab import get_data_samples_management_tab
from data_management_tab import get_checkpoint_loading_context_menu
from data_management_tab import tensor_to_image


from flask.logging import default_handler



# from cifar_exp import get_exp
# from mnist_exp import get_exp
# from imagenet_exp import get_exp
from skin_cancer_exp import get_exp

_logger = logging.getLogger("WeightsLabUI")
_logger.setLevel(logging.DEBUG)

th.manual_seed(1337)


if __name__ == '__main__':
    _logger.info("Starting server...")
    print("Starting server...")
    exp = get_exp()
    # Layout and app initialization
    app = dash.Dash(
        __name__,
        # server=False,
        external_stylesheets=[dbc.themes.ZEPHYR])
    
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    log.removeHandler(default_handler)


    def start_training_thread():
        while True:
            # print(f"[thread] while {str(exp)} at {time.time()}")
            if exp.get_is_training():
                exp.train_step_or_eval_full()
            else:
                time.sleep(0.1)


    training_thread = Thread(target=start_training_thread)
    training_thread.start()

    COLORS = defaultdict(
        lambda: f"rgb({r.randint(0,128)},{r.randint(0,128)},{r.randint(0,128)})"
    )

    checkpoint_id_to_load = None


    app.layout = html.Div([
        html.H1("GrayBox: Interactive experimentation",
                style={'textAlign': 'center'}),
        html.Div(id='last-right-clicked-id'),
        get_hyper_params_div(exp),
        get_model_architecture_tab(exp),
        get_plots_div(graph_names=[]),
        get_data_samples_management_tab(exp),
        get_layer_context_menu(),
        get_neuron_context_menu(),
        get_checkpoint_loading_context_menu(),
        dcc.Interval(id='heartbeat', interval=5*1000, n_intervals=0),
        dcc.Interval(id='refreshes', interval=2500, n_intervals=0),
        dcc.Interval(id='data-table-refresh', interval=10000, n_intervals=0),
        html.Script(src='./assets/context_menus.js'),
        html.Link(rel='stylesheet', href='./assets/context_menu.css'),
    ])


    @app.callback(
        Output('data-management-header', 'children', allow_duplicate=True),
        Output('alert-query-error', "is_open"),
        Input('run-data-query', 'n_clicks'),
        State('ad-hoc-query-input', 'value'),
        State('data-management-header', 'children'),
        prevent_initial_call=True,
    )
    def run_query_when_button_pressed(n_clicks, query, data):
        if not query:
            return data, False
        try:
            func_code = \
                "global denylist_fn\ndef denylist_fn(sample_id, age, loss, " + \
                "times, denied, prediction, label):\n\treturn " + query
            exec(func_code)
            exp.train_loader.dataset.deny_samples_with_predicate(denylist_fn)
        except Exception as e:
            print("Error running query:", query, str(e))
            return data, True

        return exp.train_loader.dataset.as_records(), False

    @app.callback(
        Output('train-tbl', 'data'),
        Input('data-table-refresh', 'n_intervals'),
    )
    def update_data_table(n_intervals):
        del n_intervals
        return exp.get_train_records()

    @app.callback(
        Output('data-management-header', 'children', allow_duplicate=True),
        Input('data-table-refresh', 'n_intervals'),
        prevent_initial_call=True
    )
    def update_number_of_data_samples(n_intervals):
        del n_intervals
        num_samples = len(exp.train_loader.dataset)
        return f'Data samples management #{num_samples}'

    @app.callback(
        Output('model-representation', 'children'),
        Input('model-representation', 'children'),
        Input('refreshes', 'n_intervals'),
        State('checkboxes', 'value')
    )
    def update_architecture_stats(previous_children, _, checklist):
        children = []
        for layer_idx, layer in enumerate(exp.model.layers):
            children.append(
                get_layer_representation(layer_idx, layer, checklist))
        return children

    @app.callback(
        Output("graphs-row-div", "children"),
        Input("heartbeat", "n_intervals"),
        State("graphs-row-div", "children")
    )
    def add_graphs_to_div(intervals, existing_children):
        if existing_children:
            return existing_children
        graph_names = exp.logger.graph_n_lines["graph_name"].unique()
        if len(graph_names) == 0:
            raise PreventUpdate
        graph_names = sorted(graph_names)
        graph_divs = []
        for graph_name in graph_names:
            graph_divs.append(dcc.Graph(
                id={"type": "graph", "index": graph_name}
            ))
        return graph_divs

    @app.callback(
        Output('resume-pause-train-btn', 'children', allow_duplicate=True),
        Input('resume-pause-train-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def resume_or_pause_train(start_clicks):
        del start_clicks
        new_children = None
        # print("[UI]pause/play button pressed ", exp, end=" ")
        if exp.get_is_training():
            new_children = get_play_button_html_elements()
        else:
            new_children = get_pause_button_html_elements()
        exp.toggle_training_status()
        # print("AFTER  ", exp)
        return new_children

    @app.callback(
        Output('resume-pause-train-btn', 'children', allow_duplicate=True),
        Input("heartbeat", "n_intervals"),
        prevent_initial_call=True,
    )
    def update_play_pause_button_icon(n_intervals):
        del n_intervals
        new_children = None
        print("[UI]update_play_pause_button_icon ", exp, end=" ")
        if exp.get_is_training():
            new_children = get_pause_button_html_elements()
        else:
            new_children = get_play_button_html_elements()
        print("AFTER  ", exp)
        return new_children

    @app.callback(
        Output("training_steps", "value", allow_duplicate=True),
        Input("heartbeat", "n_intervals"),
        prevent_initial_call=True,
    )
    def update_training_steps_input(n_intervals):
        del n_intervals
        print("[UI]update_training_steps_input ", exp)
        return exp.get_training_steps_to_do()

    @app.callback(
        Output({'type': "graph", "index": MATCH}, "figure", allow_duplicate=True),
        Input("heartbeat", "n_intervals"),
        State({'type': "graph", "index": MATCH}, "id"),
        State({'type': "graph", "index": MATCH}, "figure"),
        prevent_initial_call=True,
    )
    def update_graph(n_intervals, id, figure):
        df = exp.logger.graph_n_lines
        line_df = df[(df.graph_name == id["index"])]
        annt_df = exp.logger.lines_2_annot[
            (exp.logger.lines_2_annot.graph_name == id["index"])]

        current_model_age = exp.model.get_age()
        current_head_line_df = line_df[line_df["line_name"] == exp.name]
        closest_match = \
            (current_head_line_df["step"] - current_model_age).abs().argsort()

        if closest_match.size <= 0:
            return no_update
        current_model_metric_value = current_head_line_df.iloc[
            closest_match.iloc[0]]["line_value"]
        current_model_age = current_head_line_df.iloc[
            closest_match.iloc[0]]["step"]

        curr_plot = go.Scattergl(
            x=[current_model_age],
            y=[current_model_metric_value],
            mode='markers',
            name="HEAD",
            marker_symbol="star-diamond-open-dot",
            marker=dict(color='red', size=16)
        )
        select = go.Scattergl(
            x=[None],
            y=[None],
            mode='markers',
            name="",
            marker_symbol="diamond",
            marker=dict(color='cyan', size=12, opacity=0.8)
        )

        data = []
        for column in df.line_name.unique():
            color = COLORS[id["index"]+column]
            experiment_plot = go.Scattergl(
                x=line_df[(line_df.line_name == column)]["step"],
                y=line_df[(line_df.line_name == column)]["line_value"],
                mode="lines",
                name=column,
                line=dict(color=color)
            )
            # print("dir(scatter_plot)", dir(experiment_plot.line))
            data.append(experiment_plot)
            checkpoint_plot = go.Scattergl(
                x=annt_df[annt_df.line_name == column]["step"],
                y=annt_df[annt_df.line_name == column]["line_value"],
                mode='markers',
                marker_symbol="diamond",
                name="checkpoints-"+column,
                customdata=annt_df[annt_df.line_name == column]["metadata"],
                marker=dict(color=color, size=10))
            data.append(checkpoint_plot)

        plots = {
            'data': data + [curr_plot, select],
            'layout': go.Layout(
                title=id["index"],
                xaxis={'title': 'Steps'},
                yaxis={'title': "Value"},
            )
        }
        return plots


    @app.callback(
        Output({'type': "graph", "index": MATCH}, "figure", allow_duplicate=True),
        [
            Input({'type': "graph", "index": MATCH}, 'hoverData'),
            Input({'type': "graph", "index": MATCH}, 'clickData'),
        ],
        State({'type': "graph", "index": MATCH}, "id"),
        State({'type': "graph", "index": MATCH}, "figure"),
        prevent_initial_call=True,
    )
    def update_selection_graph(hoverData, clickData, id, figure):
        global checkpoint_id_to_load

        cursor_x = hoverData['points'][0]['x']
        cursor_y = hoverData['points'][0]['y']
        x_min, y_min, t_min, i_min, min_dist = None, None, None, None, 1e10
        for t_idx, trace_data in enumerate(figure['data']):
            if "checkpoint" not in trace_data['name']:
                continue
            x_data = np.array(trace_data['x'])
            y_data = np.array(trace_data['y'])
            if x_data.size == 0 or y_data.size == 0:
                continue

            distances = np.sqrt((x_data - cursor_x) ** 2 + (y_data - cursor_y) ** 2)
            min_index = np.argmin(distances)  # Index of the closest point

            if distances[min_index] < min_dist:
                x_min, y_min, t_min, i_min, min_dist = (
                    x_data[min_index], y_data[min_index], t_idx, min_index,
                    distances[min_index])

        if t_min is not None:
            checkpoint_id_to_load = figure['data'][t_min]["customdata"][i_min]
            figure['data'][-1]['x'] = [x_min]
            figure['data'][-1]['y'] = [y_min]

        return figure

    # This is a hacky way to pull out from the client side what was the last right
    # clicked div id. Should probabily change the name as well.
    app.clientside_callback(
        ClientsideFunction(
            namespace='clientside',
            function_name='get_text_content'
        ),
        output=Output('last-right-clicked-id', 'children'),
        inputs=[Input('refreshes', 'n_intervals'), ],
        # Use state to pass the ID of the div
        state=[Input('last-right-clicked-id', 'id')]
    )


    @app.callback(
        Input("load-checkpoint", "n_clicks"),
    )
    def print_something_on_load_checkpoint(n_clicks):
        print(f"[UI] Loading checkpopint: {checkpoint_id_to_load}")
        _logger.info(f"[UI] Loading checkpopint: {checkpoint_id_to_load}")
        exp.load(checkpoint_id=checkpoint_id_to_load)


    @app.callback(
        Input("layer-re-order", "n_clicks"),
        State('last-right-clicked-id', 'children'),
    )
    def reordering_neurons(n_clicks, children):
        print("redordering_neurons : ", n_clicks, children)
        layer_id = int(children.split("-")[-1])
        print(f"[UI] Re-Order layer {layer_id}")
        _logger.info(f"[UI] Re-Order layer {layer_id}")
        # with exp.lock:
        exp.model.reorder_neurons_by_trigger_rate(layer_id)


    @app.callback(
        Input("layer-add-neurons", "n_clicks"),
        State('last-right-clicked-id', 'children'),
    )
    def layer_add_neurons(_, children):
        layer_id = int(children.split("-")[-1])
        print(f"[UI] Add 1 neuron to layer {layer_id}")
        _logger.info(f"[UI] Add 1 neuron to layer {layer_id}")
        # with exp.lock:
        exp.model.add_neurons(layer_id=layer_id, neuron_count=1)

    @app.callback(
        [Input({"type": "layer-add-btn", "index": ALL}, "n_clicks")]
    )
    def layer_add_nuerons_btn(n_clicks):
        ctx = dash.callback_context

        if not ctx.triggered:
            return no_update

        prop_id = ctx.triggered[0]['prop_id']
        btn_dict = eval(prop_id.split('.')[0])
        layer_id = btn_dict['index']

        print(f"[UI] Add 1 neuron to layer {layer_id}")
        exp.model.add_neurons(layer_id=layer_id, neuron_count=1)

    @app.callback(
        Input("layer-reset-stats", "n_clicks"),
        State('last-right-clicked-id', 'children'),
    )
    def reset_statistics(_, children):
        print(f"[UI] RESET STATISTICS ", children, children.split("-"))
        layer_id = int(children.split("-")[-1])
        print(f"[UI] Reset statistics layer {layer_id}")
        _logger.info(f"[UI] Reset statistics layer {layer_id}")
        exp.model.reset_stats_by_layer_id(layer_id=layer_id)

    @app.callback(
        Input("neuron-re-init", "n_clicks"),
        State('last-right-clicked-id', 'children'),
    )
    def reinitializing_neurons(_, children):
        print(f"[UI] reinitializing ", children)
        layer_id, neuron_idx = [int(num) for num in re.findall(r'\d+', children)]
        print(f"[UI] Re-Init layer {layer_id} neuron {neuron_idx}")
        _logger.info(f"[UI] Re-Init layer {layer_id} neuron {neuron_idx}")
        exp.model.reinit_neurons(layer_id=layer_id, neuron_indices={neuron_idx})


    @app.callback(
        Input("neuron-prune", "n_clicks"),
        State('last-right-clicked-id', 'children'),
    )
    def removing_neurons(_, children):
        layer_id, neuron_idx = [int(num) for num in re.findall(r'\d+', children)]
        print(f"[UI] Remove layer {layer_id} neuron {neuron_idx}")
        _logger.info(f"[UI] Remove layer {layer_id} neuron {neuron_idx}")
        exp.model.prune(layer_id=layer_id, neuron_indices={neuron_idx})

    @app.callback(
        [Input({"type": "layer-rem-btn", "index": ALL}, "n_clicks")]
    )
    def remove_neurons_by_btn(n_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update

        prop_id = ctx.triggered[0]['prop_id']
        btn_dict = eval(prop_id.split('.')[0])
        layer_id = btn_dict['index']

        print(f"[UI] Prune 1 neuron from layer {layer_id}")
        layer = exp.model.get_layer_by_id(layer_id)

        exp.model.prune(
            layer_id=layer_id,
            neuron_indices={layer.neuron_count - 1})
    
    @app.callback(
        [Input({
            "type": "neuron-frozen-switch",
            "layer": ALL, "neuron": ALL
            }, "on")]
    )
    def freeze_neuron(on):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update

        prop_id = ctx.triggered[0]['prop_id']
        dct = eval(prop_id.split('.')[0])

        layer_id = int(dct['layer'])
        neuron_idx = int(dct['neuron'])

        print(
            f"[UI] Freeze/Unfreeze {on} neuron {neuron_idx} "
            f"from layer {layer_id}")
        layer = exp.model.get_layer_by_id(layer_id)

        curr_lr = layer.get_per_neuron_learning_rate(neuron_idx)

        layer.set_per_neuron_learning_rate(
            neuron_ids={neuron_idx}, lr=1.0 - curr_lr)
    
    @app.callback(
        Output('image-container', 'children'),
        Input('train-tbl', 'selected_rows'),
        State('train-tbl', 'data')
    )
    def display_image(selected_rows, data):
        if selected_rows is None or len(selected_rows) == 0:
            return no_update
        
        # Get the selected row's data
        selected_row_index = selected_rows[0]
        
        row = data[selected_row_index]
        image, id, label = exp.train_loader.dataset[selected_row_index]
        print("Data-Table row selected ", selected_row_index, row, image.shape, image)
        # encoded_image = tensor_to_image(image)
        image = th.rand(3, 28, 28)
        return html.Img(
            src=f'data:image/png;base64,{tensor_to_image(image)}',
            style={'width': '15%', 'height': 'auto'}
        )

    @app.callback(
        Input("experiment_name", "value"),
    )
    def update_experiment_name(name):
        print(f"[UI] Update experiment name: {name}")
        _logger.info(f"[UI] Update experiment name: {name}")
        exp.name = name
        return no_update


    @app.callback(
        Input("training_steps", "value"),
    )
    def update_training_steps(steps):
        print(f"[UI] Update training steps: {steps}")
        _logger.info(f"[UI] Update training steps: {steps}")
        exp.set_training_steps_to_do(steps)
        return no_update


    @app.callback(
        Input("learning_rate", "value"),
    )
    def update_learning_rate(lr):
        print(f"[UI] Update learning rate: {lr}")
        _logger.info(f"[UI] Update learning rate: {lr}")
        exp.set_learning_rate(lr)
        return no_update

    # @app.callback(
    #     Output("learning_rate", "value", allow_duplicate=True),
    #     Input("heartbeat", "n_intervals"),
    #     prevent_initial_call=True,
    # )
    # def update_learning_rate_input(n_intervals):
    #     del n_intervals
    #     return exp.learning_rate


    @app.callback(
        Input("batch_size", "value"),
    )
    def update_batch_size(batch_size):
        print(f"[UI] Update batch size: {batch_size}")
        _logger.info(f"[UI] Update batch sioze: {batch_size}")
        exp.set_batch_size(batch_size)
        return no_update

    # @app.callback(
    #     Output("batch_size", "value", allow_duplicate=True),
    #     Input("heartbeat", "n_intervals"),
    #     prevent_initial_call=True,
    # )
    # def update_training_steps_input(n_intervals):
    #     del n_intervals
    #     return exp.batch_size


    @app.callback(
        Input("eval_freq", "value"),
    )
    def update_eval_freq(eval_freq):
        exp.eval_full_to_train_steps_ratio = eval_freq
        return no_update

    # @app.callback(
    #     Output("eval_freq", "value", allow_duplicate=True),
    #     Input("heartbeat", "n_intervals"),
    #     prevent_initial_call=True,
    # )
    # def update_training_steps_input(n_intervals):
    #     del n_intervals
    #     return exp.eval_full_to_train_steps_ratio

    @app.callback(
        Input("ckpt_freq", "value"),
    )
    def update_checkpoint_freq(checkpoint_freq):
        exp.experiment_dump_to_train_steps_ratio = checkpoint_freq
        return no_update

    @app.callback(
        Output("ckpt_freq", "value", allow_duplicate=True),
        Input("heartbeat", "n_intervals"),
        prevent_initial_call=True,
    )
    def update_training_steps_input(n_intervals):
        del n_intervals
        return exp.experiment_dump_to_train_steps_ratio

    app.run(debug=False, port=8051)
    _logger.info("Stoping server...")
