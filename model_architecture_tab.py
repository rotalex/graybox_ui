"""Logic that generates the model architecture tab of the UI."""

from prettyfy import hrf
from dash import dcc
from dash import html

import dash_daq as daq
import dash_bootstrap_components as dbc

from graybox.modules_with_ops import Conv2dWithNeuronOps


def get_neuron_context_menu():
    return html.Div(
        id='neuron-context-menu',
        children=[
            html.Ul([
                html.Li('Reinitialize', id='neuron-re-init'),
                html.Li('Remove', id='neuron-prune'),
                html.Li('Freeze', id='neuron-freeze'),
            ])
        ],
        style={'display': 'none'}
    )


def get_layer_context_menu():
    return html.Div(
        id='layer-context-menu',
        children=[
            html.Ul([
                html.Li('Reorder by stats', id='layer-re-order'),
                html.Li('Add neurons', id='layer-add-neurons'),
                html.Li('Reinitialize', id='layer-re-init'),
                html.Li('Set Learning Rate', id='layer-set-lr'),
                html.Li('Reset Statistics', id='layer-reset-stats'),
            ])
        ],
        style={'display': 'none'},
    )


def get_model_representation_panel_checkbox():
    checklist = dcc.Checklist(
        id='checkboxes',
        options=[
            {'label': 'Neuron Age', 'value': 'neuron_age'},
            {'label': 'Train Trigger Rate ', 'value': 'trigger_rate_train'},
            {'label': 'Eval Trigger Rate', 'value': 'trigger_rate_eval'},
            {'label': 'Absolute diff Eval vs Train', 'value': 'abs_diff'},
            {'label': 'Relative diff Eval vs Train', 'value': 'rel_diff'},
            {'label': 'Frozen', 'value': 'frozen'},
            # {'label': 'Ignore', 'value': 'ignore'},
        ],
        value=['neuron_age', 'trigger_rate_train'],
        inline=True,
        labelStyle={'margin-right': '10px'},
    )
    return dbc.Col([checklist],)


def get_neuron_representation_heading(checklist_values):
    heading = []
    for checklist_value in checklist_values:
        if checklist_value == 'neuron_age':
            heading.append("Neuron Age")
        if checklist_value == 'trigger_rate_train':
            heading.append("Rate Train")
        if checklist_value == 'trigger_rate_eval':
            heading.append("Rate Eval")
        if checklist_value == 'abs_diff':
            heading.append("Abs Diff")
        if checklist_value == 'rel_diff':
            heading.append("Rel Diff")

    for checklist_value in checklist_values:
        if checklist_value == 'frozen':
            heading.append("Frozen")
        if checklist_value == 'ignore':
            heading.append("Ignore")

    # return html.H4(
    #     " | ".join(heading),
    #     style={
    #         'textAlign': 'center',
    #         'margin': '5px',
    #         'padding': '5px',
    #         'fontSize': '16px',
    #     }
    # )
    return dbc.Row(
        [dbc.Col(h) for h in heading],
        style={
            'color': '#DDD',
            "fontWeight": "bold",
            "fontFamily": "Monospace",
            'padding': '2px',
            'margin': '2px',
            'borderRadius': '10px',
            'fontSize': '12px',
        }
    )


def get_neuron_stats(layer, neuron_idx, checklist_values=[]):
    hts = layer.train_dataset_tracker.get_neuron_triggers(neuron_idx)
    age = layer.train_dataset_tracker.get_neuron_age(neuron_idx)
    age = max(age, 0.001)
    rate = hts / age
    age_str = hrf(age) # "%10d" % age
    rate_str = hrf(rate) # "%10.4f" % rate

    ehts = layer.eval_dataset_tracker.get_neuron_triggers(neuron_idx)
    eage = layer.eval_dataset_tracker.get_neuron_age(neuron_idx)
    eage = max(eage, 0.001)
    erate = ehts / eage
    erate_str = hrf(erate) # "%4.4f" % erate

    abs_diff = abs(rate - erate)
    rel_diff = abs_diff / max(abs(rate), 0.0001)

    values = []
    for checklist_value in checklist_values:
        if checklist_value == 'neuron_age':
            values.append(f"{age_str: <10}")
        if checklist_value == 'trigger_rate_train':
            values.append(f"{rate_str: <10}")
        if checklist_value == 'trigger_rate_eval':
            values.append(f"{erate_str: <10}")
        if checklist_value == 'abs_diff':
            str_val = hrf(abs_diff)
            values.append(f"{str_val: <10}")
        if checklist_value == 'rel_diff':
            str_val = "%4.4f" % rel_diff
            values.append(f"{str_val: <10}")
    # return str(neuron_idx) + ": " + (" | ".join(values))
    index_str = f"{neuron_idx: <4}"
    return [dbc.Col(index_str)] + [dbc.Col(value) for value in values]


def get_minus_neurons_button(layer_idx, layer):
    button = dbc.Button(
        "-",
        id={"type": "layer-rem-btn", "index": layer.get_module_id()},
        color='transparent',
        n_clicks=0,
        style={
            'fontSize': '50px',
            "borderColor": "transparent",
            "color": "red",
            'width': '5vw',
            'height': '8vh'
        }
    )
    return button


def get_plus_neurons_button(layer_idx, layer):
    button = dbc.Button(
        "+",
        id={"type": "layer-add-btn", "index": layer.get_module_id()},
        color='transparent',
        n_clicks=0,
        style={
            'fontSize': '50px',
            "borderColor": "transparent",
            "color": "green",
            'width': '5vw',
            'height': '8vh'
        }
    )
    return button


def get_plus_and_minus_buttons(layer_idx, layer):
    button_minus = get_minus_neurons_button(layer_idx, layer)
    button_plus = get_plus_neurons_button(layer_idx, layer)
    return dbc.Col([button_minus, button_plus])


def get_neuron_line_representation(
        layer_id, layer, neuron_idx, checklist_values):
    div_children = get_neuron_stats(layer, neuron_idx, checklist_values)
    # div_children.append(
    #     dbc.Col([get_neuron_stats(layer, neuron_idx, checklist_values)]))
    for checklist_value in checklist_values:
        if checklist_value == 'frozen':
            lr = layer.get_per_neuron_learning_rate(neuron_idx)
            div_children.append(
                dbc.Col([
                    daq.BooleanSwitch(
                        id={
                            'type': 'neuron-frozen-switch',
                            'layer': layer_id,
                            'neuron': neuron_idx,
                        },
                        on=lr==0,
                    )
                ],
                style={'width': 'auto'})
            )
        if checklist_value == 'ignore':
            div_children.append(
                dbc.Col([
                    daq.BooleanSwitch(
                        id={
                            'type': 'neuron-ignore-switch',
                            'layer': layer_id,
                            'neuron': neuron_idx,
                        },
                        on=False,
                    )
                ],
                style={'width': 'auto'})
            )

    return html.Div(
        id=f"layer{layer_id}neuron{neuron_idx}",
        children=dbc.Row(id=f"layerow{layer_id}neuron{neuron_idx}", children=div_children),
        style={
            "color": "black",
            "fontWeight": "bold",
            "fontFamily": "Monospace",
            'padding': '2px',
            'backgroundColor': '#DDD',
            'margin': '2px',
            'borderRadius': '10px',
            'fontSize': '12px',
            # 'width': '16vw',
            # 'min-width': '15vw',
        }
    )


def get_layer_representation(layer_idx, layer, checklist_values):
    if not hasattr(layer, 'train_dataset_tracker'):
        return None

    layer_name = "Conv" if isinstance(layer, Conv2dWithNeuronOps) else "Linear"
    kernel_rpr = ""
    if layer_name == "Conv":
        kernel_rpr = f"{layer.weight.shape[2]}x{layer.weight.shape[3]}->"
    layer_name += f"[{layer.incoming_neuron_count}->{kernel_rpr}{layer.neuron_count}]"

    layer_id = str(layer.get_module_id())
    return html.Div(
        id="layer-representation-" + str(layer.get_module_id()),
        children=[
            html.H3(layer_name, style={'textAlign': 'center'}),
            get_neuron_representation_heading(checklist_values),
            html.Div(
                id=layer_id,
                className="layer-representation",
                children=[
                    get_neuron_line_representation(
                            layer_id, layer, neuron_idx, checklist_values) \
                    for neuron_idx in range(
                        layer.train_dataset_tracker.number_of_neurons)
                ],
                style={
                    'overflowY': 'auto',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center',
                    'height': '33vh',
                }
            ),
            get_plus_and_minus_buttons(layer_idx, layer),
        ],
        style={
            'margin': '15px',
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
            'border': '2px solid #666',
            'padding': '5px',
            'borderRadius': '15px',
            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
            # 'max-width': '33vw',
            'min-width': '30vw',
        }
    )


def get_model_representation_div(
        exp: "Experiment",
        checklist_values=["neuron_age", "neuron_trigger_rate_train"]
    ):
    children = []
    for layer_idx, layer in enumerate(exp.model.layers):
        children.append(
            get_layer_representation(layer_idx, layer, checklist_values))

    return html.Div(
        id="model-representation",
        children=children,
        style={
            'display': 'flex',
            'justifyContent': 'space-around',
            'alignItems': 'center',
            'overflowX': 'auto',
        }
    )


def get_model_architecture_tab(exp: "Experiment"):
    return html.Div(
        id="model-architecture-tab",
        children=[
            get_model_representation_panel_checkbox(),
            get_model_representation_div(exp)
        ],
    )