from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from explanation.tools import utils as ut
from explanation.explainer import SurvExplainer
from src.components import ids


def render(app: Dash, explainer: SurvExplainer) -> html.Div:

    @app.callback(
        Output(ids.CPP_GRAPH, "children"),
        Input(ids.OBSERVATION_DROPDOWN, "value"),
        Input(ids.FEATURE_DROPDOWN, 'value')
    )
    def update_cpp_graph(new_observation_id: int, feature: str) -> html.Div:
        fig = explainer \
            .cp_profile(feature, ut.make_single_observation_by_id(explainer.X, new_observation_id)) \
            .plot()
        return html.Div(dcc.Graph(figure=fig), id=ids.CPP_GRAPH)

    return html.Div(id=ids.CPP_GRAPH)
