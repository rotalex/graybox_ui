"""Generate plots related div."""
from dash import dcc

import dash_bootstrap_components as dbc
import plotly.graph_objs as go


def get_plots_div(graph_names=[]):
    cols = []
    for graph_name in graph_names:
        cols.append(
            dcc.Graph(
                id=f"{graph_name}-graph",
                config={"displayModeBar": False},
                figure=go.FigureWidget()),
        )
    return dbc.Row(cols, id="graphs-row-div")