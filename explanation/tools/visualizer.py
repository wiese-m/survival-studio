import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class Visualizer:
    def __init__(self, model, X: pd.DataFrame, observation: pd.DataFrame) -> None:
        self.model = model
        self.X = X
        self.observation = observation

    def plot_surv(self, observation: pd.DataFrame = None, show: bool = False, **kwargs) -> go.Figure:
        observation = self.observation if observation is None else observation
        surv = self.model.predict_survival_function(observation)
        fig = px.line(x=surv[0].x, y=surv[0].y, line_shape='hv')
        fig.update_xaxes(title_text='time')
        fig.update_yaxes(title_text='survival probability')
        fig.update_layout(yaxis_range=[0, 1])
        fig.update_layout(**kwargs)
        if show:
            fig.show()
        return fig

    def plot_chf(self, observation: pd.DataFrame = None, show: bool = False, **kwargs) -> go.Figure:
        observation = self.observation if observation is None else observation
        chf = self.model.predict_cumulative_hazard_function(observation)
        fig = px.line(x=chf[0].x, y=chf[0].y, line_shape='hv')
        fig.update_xaxes(title_text='time')
        fig.update_yaxes(title_text='cumulative hazard')
        fig.update_layout(**kwargs)
        if show:
            fig.show()
        return fig

    def plot_feature_distribution(self, feature: str, nbins: int = None, show: bool = False, **kwargs) -> go.Figure:
        fig = px.histogram(self.X, x=feature, nbins=nbins)
        fig.update_layout(title_text='Feature Distribution', **kwargs)
        if show:
            fig.show()
        return fig
