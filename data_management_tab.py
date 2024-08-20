"""Create data management tab for the experiment."""

from dash import dcc
from dash import dash_table
from dash import html
import dash_bootstrap_components as dbc


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