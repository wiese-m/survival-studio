import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class PartialDependence:
    def __init__(self, model, X: pd.DataFrame, feature: str) -> None:
        self.feature = feature
        self.model = model
        self.X = X.copy()
        self.result = self._get_result()

    @property
    def mean_prediction(self) -> float:
        # Get the mean prediction of the model on the input data
        return self.model.predict(self.X).mean()

    def fit(self, value) -> None:
        # Update input data with given value of the feature
        self.X[self.feature] = value

    def _prepare_values(self) -> np.ndarray:
        # Prepare a sequence of values to use for computing partial dependence
        if self.X[self.feature].dtype in (np.int64, int):
            min_, max_ = self.X[self.feature].min(), self.X[self.feature].max()
            return np.unique(np.linspace(min_, max_, 100, dtype=int))
        elif self.X[self.feature].dtype in (np.float64, float):
            min_, max_ = self.X[self.feature].min(), self.X[self.feature].max()
            return np.linspace(min_, max_, 100)
        return self.X[self.feature].unique()

    def _get_result(self) -> pd.DataFrame:
        # Compute the average prediction for each prepared value of the feature
        data = {self.feature: [], 'avg_risk_score': []}
        for value in self._prepare_values():
            self.fit(value)
            data[self.feature].append(value)
            data['avg_risk_score'].append(self.mean_prediction)
        return pd.DataFrame(data).sort_values(self.feature).reset_index(drop=True)

    def plot(self, is_categorical: bool = False, show: bool = False, **kwargs) -> go.Figure:
        # Generate a plot of the partial dependence results
        if not is_categorical:
            fig = px.line(x=self.result[self.feature], y=self.result['avg_risk_score'])
            fig.update_xaxes(title_text=self.feature)
            fig.update_yaxes(title_text='avg risk score')
        # Generate a bar plot of the partial dependence results for categorical feature
        else:
            fig = px.bar(y=self.result[self.feature], x=self.result['avg_risk_score'], orientation='h')
            fig.update_yaxes(title_text=self.feature)
            fig.update_xaxes(title_text='avg risk score')
        fig.update_layout(title_text=f'Partial Dependence', **kwargs)
        if show:
            fig.show()
        return fig
