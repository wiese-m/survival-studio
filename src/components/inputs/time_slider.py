from dash import dcc, html

from explanation.explainer import SurvExplainer
from src.components import ids


def render(explainer: SurvExplainer) -> html.Div:
    event_times = explainer.model.event_times_

    return html.Div(
        children=[
            html.H6('Time Slider for Integrated Brier Score'),
            dcc.RangeSlider(
                event_times.min(),
                event_times.max(),
                value=[event_times.min(), event_times.max()],
                id=ids.TIME_SLIDER,
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ]
    )
