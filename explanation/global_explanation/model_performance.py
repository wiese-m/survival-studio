import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sksurv.metrics import (
    concordance_index_ipcw,
    brier_score,
    integrated_brier_score
)


class ModelPerformance:
    def __init__(self, model, X_test: pd.DataFrame, y_test: np.ndarray) -> None:
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.survs = model.predict_survival_function(X_test)  # todo: uwzglednic SSVM itp.
        self.event_times = self._event_times

    def harrell_cindex(self, X: pd.DataFrame = None, y: np.ndarray = None) -> float:
        X = X if X is not None else self.X_test
        y = y if y is not None else self.y_test
        return self.model.score(X, y)

    def uno_cindex(self, y_train: np.ndarray = None, y_test: np.ndarray = None,
                   tau: float = None, tied_tol: float = 1e-8) -> float:
        y_train = y_train if y_train is not None else self.y_test
        y_test = y_test if y_test is not None else self.y_test
        cindex, concordant, discordant, tied_risk, tied_time = \
            concordance_index_ipcw(y_train, y_test, self.model.predict(self.X_test), tau, tied_tol)
        return cindex

    def brier_score(self, time, y_train: np.ndarray = None, y_test: np.ndarray = None) -> float:
        y_train = y_train if y_train is not None else self.y_test
        y_test = y_test if y_test is not None else self.y_test
        preds = [surv_func(time) for surv_func in self.survs]
        times, score = brier_score(y_train, y_test, preds, time)
        return score[0]

    def integrated_brier_score(self, times=None, y_train: np.ndarray = None, y_test: np.ndarray = None) -> float:
        y_train = y_train if y_train is not None else self.y_test
        y_test = y_test if y_test is not None else self.y_test
        times = times if times is not None else self.event_times
        preds = np.asarray([[surv_func(t) for t in times] for surv_func in self.survs])
        return integrated_brier_score(y_train, y_test, preds, times)

    def _get_bs_plot_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.event_times, columns=['event_time'])
        df['brier_score'] = [self.brier_score(t) for t in df.event_time]
        return df

    def plot_brier_score(self, show: bool = False) -> go.Figure:
        fig = px.line(data_frame=self._get_bs_plot_df(), x='event_time', y='brier_score')
        fig.update_xaxes(title_text='time')
        fig.update_yaxes(title_text='brier score')
        fig.update_layout(width=400, height=300)
        if show:
            fig.show()
        return fig

    @property
    def _event_times(self) -> np.ndarray:
        return self.survs[0].x
