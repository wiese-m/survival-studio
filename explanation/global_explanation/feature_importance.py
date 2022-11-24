import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from eli5.sklearn import PermutationImportance


class FeatureImportance(PermutationImportance):
    def __init__(self, model, X: pd.DataFrame, y: np.ndarray, n_iter: int = 5, random_state: int = None) -> None:
        super().__init__(model, n_iter=n_iter, random_state=random_state)
        self.result = self._get_result(X, y)

    def _get_result(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        self.fit(X, y)
        return pd.DataFrame(self.results_, columns=X.columns)

    def plot(self, show: bool = False, **kwargs) -> go.Figure:
        result_long = pd.melt(self.result, var_name='feature', value_name='value')
        ordering = np.mean(self.result, axis=0).sort_values().index
        fig = px.box(result_long, x='value', y='feature')
        fig.update_yaxes(title_text='', categoryorder='array', categoryarray=ordering)
        fig.update_xaxes(title_text="Harrell's c-index decrease")
        fig.update_traces(boxmean=True)
        fig.update_layout(title_text='Permutation Feature Importance', **kwargs)
        if show:
            fig.show()
        return fig

    # def _get_result(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    #     self.fit(X, y)
    #     return pd.DataFrame(
    #         {'feature_name': X.columns,
    #          'feature_importance': self.feature_importances_,
    #          'fi_std': self.feature_importances_std_,
    #          'fi_2std': 2 * self.feature_importances_std_}
    #     ).sort_values('feature_importance', ascending=False)
    #
    # def plot(self, show: bool = False, **kwargs) -> go.Figure:
    #     fig = px.bar(self.result[::-1], x='feature_importance', y='feature_name', orientation='h', error_x='fi_2std')
    #     fig.update_xaxes(title_text='harrell c-index decrease')
    #     fig.update_yaxes(title_text='')
    #     fig.update_layout(title_text='Permutation Feature Importance', **kwargs)
    #     if show:
    #         fig.show()
    #     return fig
