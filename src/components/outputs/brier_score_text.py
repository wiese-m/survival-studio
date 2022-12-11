from dash import Dash, html
from dash.dependencies import Input, Output

from explanation.explainer import SurvExplainer
from src.components import ids


def render(app: Dash, explainer: SurvExplainer) -> html.Div:

    @app.callback(
        Output(ids.BRIER_SCORE, 'children'),
        Input(ids.TIME_INPUT, 'value')
    )
    def update_brier_score(time) -> str:
        if not explainer.model_performance.can_predict_survival():
            return ''
        bs = explainer.model_performance.brier_score(time)
        return f'BS({time}) = {bs:.4f}'

    return html.Div(id=ids.BRIER_SCORE)
