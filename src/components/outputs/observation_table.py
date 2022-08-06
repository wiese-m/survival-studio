from dash import Dash, html
from dash.dependencies import Input, Output

from explanation.tools import utils as ut
from explanation.explainer import SurvExplainer
from src.components import ids


def render(app: Dash, explainer: SurvExplainer) -> html.Div:

    @app.callback(
        Output(ids.OBSERVATION_TABLE, "children"),
        Input(ids.OBSERVATION_DROPDOWN, "value")
    )
    def update_observation_table(new_observation_id: int) -> html.Div:
        children = [
            ut.generate_table(ut.make_single_observation_by_id(explainer.X, new_observation_id))
        ]
        return html.Div(children=children, id=ids.OBSERVATION_TABLE)

    return html.Div(id=ids.OBSERVATION_TABLE)
