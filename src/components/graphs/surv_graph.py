from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from explanation.tools import utils as ut
from explanation.explainer import SurvExplainer
from src.components import ids


def render(app: Dash, explainer: SurvExplainer) -> html.Div:

    @app.callback(
        Output(ids.SURV_GRAPH, "children"),
        Input(ids.OBSERVATION_DROPDOWN, "value")
    )
    def update_surv_graph(new_observation_id: int) -> html.Div:
        fig = explainer.visualizer.plot_surv(ut.make_single_observation_by_id(explainer.X, new_observation_id))
        return html.Div(dcc.Graph(figure=fig), id=ids.SURV_GRAPH)

    return html.Div(id=ids.SURV_GRAPH)
