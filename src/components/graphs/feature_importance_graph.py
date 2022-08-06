from dash import dcc, html

from explanation.explainer import SurvExplainer
from src.components import ids


def render(explainer: SurvExplainer) -> html.Div:
    fig = explainer.feature_importance().plot()
    return html.Div(dcc.Graph(figure=fig), id=ids.FEATURE_IMPORTANCE_GRAPH)
