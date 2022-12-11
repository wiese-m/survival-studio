from dash import Dash, html
from dash.dependencies import Input, Output

from explanation.explainer import SurvExplainer
from src.components import ids


def render(app: Dash, explainer: SurvExplainer) -> html.Div:

    @app.callback(
        Output(ids.INTEGRATED_BRIER_SCORE, 'children'),
        Input(ids.TIME_SLIDER, 'value')
    )
    def update_integrated_brier_score(times) -> str:
        if not explainer.model_performance.can_predict_survival():
            return ''
        ibs = explainer.model_performance.integrated_brier_score(times)
        return f'IBS({times}) = {ibs:.4f}'

    return html.Div(id=ids.INTEGRATED_BRIER_SCORE)
