from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from explanation.explainer import SurvExplainer
from explanation.tools import utils as ut
from src.components import ids
from src.components.visualization_options import spec


def render(app: Dash, explainer: SurvExplainer) -> html.Div:

    @app.callback(
        Output(ids.BDP_GRAPH, "children"),
        Input(ids.OBSERVATION_DROPDOWN, "value")
    )
    def update_bdp_graph(new_observation_id: int) -> html.Div:
        fig = explainer \
            .bd_profile(
                new_observation=ut.make_single_observation_by_id(explainer.X, new_observation_id),
                allow_interactions=False  # todo: add interaction to choose
            ).plot(**spec)
        return html.Div(dcc.Graph(figure=fig), id=ids.BDP_GRAPH)

    return html.Div(id=ids.BDP_GRAPH)
