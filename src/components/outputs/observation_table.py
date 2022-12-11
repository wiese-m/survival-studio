from dash import Dash, html, dash_table
from dash.dependencies import Input, Output

from explanation.explainer import SurvExplainer
from explanation.tools import utils as ut
from src.components import ids


def render(app: Dash, explainer: SurvExplainer) -> html.Div:

    @app.callback(
        Output(ids.OBSERVATION_TABLE, "children"),
        Input(ids.OBSERVATION_DROPDOWN, "value")
    )
    def update_observation_table(new_observation_id: int) -> html.Div:
        new_observation = ut.make_single_observation_by_id(explainer.X, new_observation_id).round(2)
        children = [
            dash_table.DataTable(
                data=new_observation.to_dict('records'),
                columns=[{"name": i, "id": i} for i in new_observation.columns],
                page_size=1
            )
        ]
        return html.Div(children=children, id=ids.OBSERVATION_TABLE)

    return html.Div(id=ids.OBSERVATION_TABLE)
