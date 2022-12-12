from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from explanation.tools.utils import make_single_observation_by_id


class Visualizer:
    def __init__(self, model, X: pd.DataFrame, observation: pd.DataFrame) -> None:
        self.model = model
        self.X = X
        self.observation = observation

    # Plot survival function estimated by the model
    def plot_surv(self, observation: pd.DataFrame = None, show: bool = False, **kwargs) -> go.Figure:
        observation = self.observation if observation is None else observation
        surv = self.model.predict_survival_function(observation)
        fig = px.line(x=surv[0].x, y=surv[0].y, line_shape='hv')
        fig.update_xaxes(title_text='time')
        fig.update_yaxes(title_text='survival probability')
        fig.update_layout(title_text='Survival Function', yaxis_range=[0, 1], **kwargs)
        if show:
            fig.show()
        return fig

    # Plot cumulative hazard function estimated by the model
    def plot_chf(self, observation: pd.DataFrame = None, show: bool = False, **kwargs) -> go.Figure:
        observation = self.observation if observation is None else observation
        chf = self.model.predict_cumulative_hazard_function(observation)
        fig = px.line(x=chf[0].x, y=chf[0].y, line_shape='hv')
        fig.update_xaxes(title_text='time')
        fig.update_yaxes(title_text='cumulative hazard')
        fig.update_layout(title_text='Cumulative Hazard Function', **kwargs)
        if show:
            fig.show()
        return fig

    # Plot feature distribution (histogram) for given feature
    def plot_feature_distribution(self, feature: str, nbins: int = None, show: bool = False, **kwargs) -> go.Figure:
        fig = px.histogram(self.X, x=feature, nbins=nbins)
        fig.update_traces(marker_line_width=1, marker_line_color='black')
        fig.update_layout(title_text='Feature Distribution', **kwargs)
        if show:
            fig.show()
        return fig

    # Plot prediction distribution (histogram)
    def plot_risk_scores_distribution(self, nbins: int = None, show: bool = False, **kwargs) -> go.Figure:
        fig = px.histogram(self._get_risk_scores(), nbins=nbins)
        fig.update_xaxes(title_text='risk score')
        fig.update_traces(marker_line_width=1, marker_line_color='black')
        fig.update_layout(title_text='Risk Scores Distribution', **kwargs)
        if show:
            fig.show()
        return fig

    def _get_risk_scores(self) -> List[float]:
        return [float(self.model.predict(make_single_observation_by_id(self.X, id_))) for id_ in self.X.index]
