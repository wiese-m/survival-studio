import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
from dash import Dash, html

from explanation.explainer import SurvExplainer
from .graphs import bdp_graph, cpp_graph, feature_distribution_graph, feature_importance_graph, surv_graph, pdp_graph
from .inputs import feature_dropdown, observation_dropdown, time_input, time_slider
from .outputs import brier_score_text, integrated_brier_score_text, observation_table


def create_layout(app: Dash, explainer: SurvExplainer) -> html.Div:
    return html.Div(
        [
            dbc.Row(
                dbc.Col(html.H1(f'{app.title} ({explainer.model_name})'))
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(observation_dropdown.render(explainer)),
                            html.Div(feature_dropdown.render(explainer))
                        ]
                    ),
                    dbc.Col(
                        [
                            html.Div(time_input.render(explainer)),
                            html.Div(time_slider.render(explainer))
                        ]
                    ),
                    dbc.Col(
                        [
                            html.H5(harrell_cindex_text(explainer)),
                            html.H5(uno_cindex_text(explainer)),
                            html.H5(brier_score_text.render(app, explainer)),
                            html.H5(integrated_brier_score_text.render(app, explainer))
                        ]
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dls.Hash(surv_graph.render(app, explainer))),
                    dbc.Col(dls.Hash(cpp_graph.render(app, explainer))),
                    dbc.Col(dls.Hash(bdp_graph.render(app, explainer)))
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dls.Hash(feature_distribution_graph.render(app, explainer))),
                    dbc.Col(dls.Hash(pdp_graph.render(app, explainer))),
                    dbc.Col(dls.Hash(feature_importance_graph.render(explainer)))
                ]
            ),
            dbc.Row(
                dbc.Col(observation_table.render(app, explainer))
            )
        ]
    )


def harrell_cindex_text(explainer: SurvExplainer) -> str:
    return f"Harrell's C-index = {explainer.model_performance.harrell_cindex(explainer.X, explainer.y):.4f}"


def uno_cindex_text(explainer: SurvExplainer) -> str:
    return f"Uno's C-index = {explainer.model_performance.uno_cindex(explainer.y, explainer.y):.4f}"
