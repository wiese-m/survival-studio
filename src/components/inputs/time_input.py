import numpy as np
from dash import dcc, html

from explanation.explainer import SurvExplainer
from src.components import ids


def render(explainer: SurvExplainer) -> html.Div:
    if not explainer.model_performance.can_predict_survival():
        return html.Div()
    times = explainer.model_performance.proper_times

    return html.Div(
        children=[
            html.H6('Time Input for Brier Score'),
            dcc.Input(
                id=ids.TIME_INPUT,
                type='number',
                placeholder=f'from {times.min()} to {times.max()}',
                min=times.min(),
                max=times.max(),
                value=np.median(times)
            )
        ]
    )
