from dash import dcc, html

from explanation.explainer import SurvExplainer
from src.components import ids
from src.components.visualization_options import spec


# Make interactive Permutation Feature Importance plot (default iterations: 10)
def render(explainer: SurvExplainer) -> html.Div:
    fig = explainer.feature_importance(n_iter=10, random_state=2022).plot(**spec)
    return html.Div(dcc.Graph(figure=fig), id=ids.FEATURE_IMPORTANCE_GRAPH)
