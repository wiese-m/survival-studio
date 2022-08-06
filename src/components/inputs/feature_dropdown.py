import random

from dash import dcc, html

from explanation.explainer import SurvExplainer
from src.components import ids


def render(explainer: SurvExplainer) -> html.Div:
    all_features = explainer.X.columns

    return html.Div(
        children=[
            html.H6("Feature"),
            dcc.Dropdown(
                id=ids.FEATURE_DROPDOWN,
                options=[{"label": feature, "value": feature} for feature in all_features],
                value=random.choice(all_features)
            )
        ]
    )
