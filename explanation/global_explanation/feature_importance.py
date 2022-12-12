import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from eli5.sklearn import PermutationImportance


class FeatureImportance(PermutationImportance):
    def __init__(self, model, X: pd.DataFrame, y: np.ndarray, n_iter: int = 5, random_state: int = None) -> None:
        super().__init__(model, n_iter=n_iter, random_state=random_state)
        self.result = self._compute_importance(X, y)

    def _compute_importance(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        # Compute the permutation feature importance using eli5 fit method
        self.fit(X, y)
        return pd.DataFrame(self.results_, columns=X.columns)

    def plot(self, show: bool = False, **kwargs) -> go.Figure:
        # Transform the results to a long-form DataFrame for box plot
        result_long = pd.melt(self.result, var_name='feature', value_name='value')
        # Create a list of the features ordered according to their mean importance
        ordering = np.mean(self.result, axis=0).sort_values().index
        # Generate a permutation feature importance box plot using plotly express
        fig = px.box(result_long, x='value', y='feature')
        fig.update_yaxes(title_text='', categoryorder='array', categoryarray=ordering)
        fig.update_xaxes(title_text="Harrell's c-index decrease")
        fig.update_traces(boxmean=True)
        fig.update_layout(title_text='Permutation Feature Importance', **kwargs)
        if show:
            fig.show()
        return fig
