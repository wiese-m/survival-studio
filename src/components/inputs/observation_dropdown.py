import random

from dash import dcc, html

from explanation.explainer import SurvExplainer
from src.components import ids


def render(explainer: SurvExplainer) -> html.Div:
    all_observations = explainer.X.index

    return html.Div(
        children=[
            html.H6("Observation"),
            dcc.Dropdown(
                id=ids.OBSERVATION_DROPDOWN,
                options=[{"label": f'id: {o}, y: {explainer.y[explainer.X.index.get_loc(o)]}', "value": o}
                         for o in all_observations],
                value=random.choice(all_observations)
            )
        ]
    )
