from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from explanation.explainer import SurvExplainer
from explanation.tools import utils as ut
from src.components import ids
from src.components.visualization_options import spec


def render(app: Dash, explainer: SurvExplainer) -> html.Div:

    @app.callback(
        Output(ids.SURV_GRAPH, "children"),
        Input(ids.OBSERVATION_DROPDOWN, "value")
    )
    def update_surv_graph(new_observation_id: int) -> html.Div:
        if not explainer.model_performance.can_predict_survival():
            return html.Div(html.H4('Chosen model cannot predict survival function.', className='text-center'))
        fig = explainer.visualizer.plot_surv(ut.make_single_observation_by_id(explainer.X, new_observation_id), **spec)
        return html.Div(dcc.Graph(figure=fig), id=ids.SURV_GRAPH)

    return html.Div(id=ids.SURV_GRAPH)
