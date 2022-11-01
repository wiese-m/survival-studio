from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from explanation.explainer import SurvExplainer
from src.components import ids
from src.components.visualization_options import spec


def render(app: Dash, explainer: SurvExplainer) -> html.Div:

    @app.callback(
        Output(ids.FEATURE_DISTRIBUTION_GRAPH, "children"),
        Input(ids.FEATURE_DROPDOWN, 'value')
    )
    def update_fd_graph(feature: str) -> html.Div:
        fig = explainer.visualizer.plot_feature_distribution(feature, **spec)
        return html.Div(dcc.Graph(figure=fig), id=ids.FEATURE_DISTRIBUTION_GRAPH)

    return html.Div(id=ids.FEATURE_DISTRIBUTION_GRAPH)
