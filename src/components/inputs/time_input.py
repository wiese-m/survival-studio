import numpy as np
from dash import dcc, html

from explanation.explainer import SurvExplainer
from src.components import ids


def render(explainer: SurvExplainer) -> html.Div:
    event_times = explainer.model.event_times_

    return html.Div(
        children=[
            html.H6('Time Input for Brier Score'),
            dcc.Input(
                id=ids.TIME_INPUT,
                type='number',
                placeholder=f'from {event_times.min()} to {event_times.max()}',
                min=event_times.min(),
                max=event_times.max(),
                value=np.median(event_times)
            )
        ]
    )
