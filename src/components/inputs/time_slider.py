from dash import dcc, html

from explanation.explainer import SurvExplainer
from src.components import ids


# Make time slider for Integrated Brier Score calculation
def render(explainer: SurvExplainer) -> html.Div:
    if not explainer.model_performance.can_predict_survival():
        return html.Div()
    times = explainer.model_performance.proper_times

    return html.Div(
        children=[
            html.H6('Time Slider for Integrated Brier Score'),
            dcc.RangeSlider(
                times.min(),
                times.max(),
                value=[times.min(), times.max()],
                id=ids.TIME_SLIDER,
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ]
    )
