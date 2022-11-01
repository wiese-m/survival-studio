from dash import dcc, html

from explanation.explainer import SurvExplainer
from src.components import ids
from src.components.visualization_options import spec


def render(explainer: SurvExplainer) -> html.Div:
    fig = explainer.feature_importance().plot(**spec)
    return html.Div(dcc.Graph(figure=fig), id=ids.FEATURE_IMPORTANCE_GRAPH)
