import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class CeterisParibus:
    def __init__(self, model, X: pd.DataFrame, new_observation: pd.DataFrame, feature: str) -> None:
        self.feature = feature
        self.model = model
        self.original_observation = new_observation
        self._new_observation = new_observation.copy()
        self.original_prediction = self.new_prediction
        self.result = self._get_result(X, feature)

    @property
    def new_prediction(self) -> float:
        return self.model.predict(self._new_observation)[0]

    def fit(self, feature: str, value) -> None:
        self._new_observation[feature] = value

    @staticmethod
    def _prepare_values(X: pd.DataFrame, feature: str) -> np.ndarray:
        if X[feature].dtype in (np.int64, int):
            min_, max_ = X[feature].min(), X[feature].max()
            return np.unique(np.linspace(min_, max_, 100, dtype=int))
        elif X[feature].dtype in (np.float64, float):
            min_, max_ = X[feature].min(), X[feature].max()
            return np.linspace(min_, max_, 100)
        return X[feature].unique()

    def _get_result(self, X: pd.DataFrame, feature: str) -> pd.DataFrame:
        data = {feature: [], 'risk_score': []}
        for value in self._prepare_values(X, feature):
            self.fit(feature, value)
            data[feature].append(value)
            data['risk_score'].append(self.new_prediction)
        return pd.DataFrame(data).sort_values(feature).reset_index(drop=True)

    def plot(self, is_categorical: bool = False, show: bool = False, **kwargs) -> go.Figure:
        if not is_categorical:
            trace0 = go.Scatter(x=self.original_observation[self.feature],
                                y=pd.Series(self.original_prediction), name='')
            trace1 = go.Scatter(x=self.result[self.feature], y=self.result['risk_score'],
                                name='', marker=dict(opacity=0))
        else:
            trace0 = go.Bar(y=self.original_observation[self.feature], x=pd.Series(self.original_prediction),
                            name='', offsetgroup=0, orientation='h')
            trace1 = go.Bar(y=self.result[self.feature], x=self.result['risk_score'],
                            name='', offsetgroup=0, orientation='h')
        fig = make_subplots()
        fig.add_trace(trace1)
        fig.add_trace(trace0)
        if not is_categorical:
            fig.update_xaxes(title_text=self.feature)
            fig.update_yaxes(title_text='risk score')
        else:
            fig.update_yaxes(title_text=self.feature)
            fig.update_xaxes(title_text='risk score')
        fig.update_layout(title_text='Ceteris Paribus', **kwargs)
        if show:
            fig.show()
        return fig
