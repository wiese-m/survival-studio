from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from explanation.explainer import SurvExplainer
from src.components import ids
from src.components.visualization_options import spec


# Make interactive Partial Dependence plot for chosen feature
def render(app: Dash, explainer: SurvExplainer) -> html.Div:

    @app.callback(
        Output(ids.PDP_GRAPH, "children"),
        Input(ids.FEATURE_DROPDOWN, 'value')
    )
    def update_pdp_graph(feature: str) -> html.Div:
        fig = explainer.pd_profile(feature).plot(**spec)
        return html.Div(dcc.Graph(figure=fig), id=ids.PDP_GRAPH)

    return html.Div(id=ids.PDP_GRAPH)
