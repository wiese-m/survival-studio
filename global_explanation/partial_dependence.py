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
        return self.model.predict(self.X).mean()

    def fit(self, value) -> None:
        self.X[self.feature] = value

    def _get_result(self) -> pd.DataFrame:
        data = {self.feature: [], 'avg_risk_score': []}
        for value in self.X[self.feature].unique():
            self.fit(value)
            data[self.feature].append(value)
            data['avg_risk_score'].append(self.mean_prediction)
        return pd.DataFrame(data).sort_values(self.feature).reset_index(drop=True)

    def plot(self, show: bool = False) -> go.Figure:
        fig = px.line(x=self.result[self.feature], y=self.result['avg_risk_score'])
        fig.update_xaxes(title_text=self.feature)
        fig.update_yaxes(title_text='avg risk score')
        fig.update_layout(title_text=f'Partial Dependence', width=600, height=400)
        if show:
            fig.show()
        return fig
