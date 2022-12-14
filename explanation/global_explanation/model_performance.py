import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sksurv.metrics import (
    concordance_index_ipcw,
    brier_score,
    integrated_brier_score
)

from explanation.tools.model_enum import SurvivalModel


class ModelPerformance:
    def __init__(self, model, X_test: pd.DataFrame, y_test: np.ndarray) -> None:
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.survs = self._get_survs()

    # Check if the model is capable of predicting survival probabilities
    def can_predict_survival(self):
        return self._model_enum.can_predict_survival() and self._check_gbm_loss()

    # Check if the model is a Gradient Boosting Machine (GBM) with 'coxph' loss function
    def _check_gbm_loss(self):
        return False if self._model_enum.is_gbm() and self.model.loss_ != 'coxph' else True

    # Compute the predicted survival function for the model (if possible)
    def _get_survs(self) -> np.ndarray:
        return self.model.predict_survival_function(self.X_test) if self.can_predict_survival() else None

    # Compute the Harrell's concordance index for the model
    # This metric measures the ability of the model to correctly rank survival times
    def harrell_cindex(self, X: pd.DataFrame = None, y: np.ndarray = None) -> float:
        X = X if X is not None else self.X_test
        y = y if y is not None else self.y_test
        return self.model.score(X, y)

    # Compute the Uno's concordance index for the model
    # This metric is similar to Harrell's C index but also accounts for the uncertainty of the predicted survival times
    def uno_cindex(self, y_train: np.ndarray = None, y_test: np.ndarray = None,
                   tau: float = None, tied_tol: float = 1e-8) -> float:
        y_train = y_train if y_train is not None else self.y_test
        y_test = y_test if y_test is not None else self.y_test
        cindex, concordant, discordant, tied_risk, tied_time = \
            concordance_index_ipcw(y_train, y_test, self.model.predict(self.X_test), tau, tied_tol)
        return cindex

    # Compute the Brier score for the model at a given time
    # The Brier score is a measure of the model's accuracy in predicting the probability of survival
    def brier_score(self, time, y_train: np.ndarray = None, y_test: np.ndarray = None) -> float:
        if self.survs is None:
            return np.nan
        y_train = y_train if y_train is not None else self.y_test
        y_test = y_test if y_test is not None else self.y_test
        preds = [surv_func(time) for surv_func in self.survs]
        times, score = brier_score(y_train, y_test, preds, time)
        return score[0]

    # Compute the integrated Brier score for the model over a given set of times
    # This is a weighted average of the Brier scores at each time
    def integrated_brier_score(self, times=None, y_train: np.ndarray = None, y_test: np.ndarray = None) -> float:
        if self.survs is None:
            return np.nan
        y_train = y_train if y_train is not None else self.y_test
        y_test = y_test if y_test is not None else self.y_test
        times = times if times is not None else self.proper_times
        preds = np.asarray([[surv_func(t) for t in times] for surv_func in self.survs])
        return integrated_brier_score(y_train, y_test, preds, times)

    # Generate a DataFrame containing the Brier scores at each time
    def _get_bs_plot_df(self) -> pd.DataFrame:
        if self.survs is None:
            return pd.DataFrame()
        min_, max_ = np.min(self.proper_times), np.max(self.proper_times)
        df = pd.DataFrame(np.unique(np.linspace(min_, max_, 1000, dtype=int)), columns=['time'])
        df['brier_score'] = [self.brier_score(t) for t in df.time]
        return df

    # Plot prediction error curve over time based on Brier score
    def plot_brier_score(self, show: bool = False, **kwargs) -> go.Figure:
        plot_df = self._get_bs_plot_df()
        if plot_df.empty:
            return go.Figure()
        fig = px.line(data_frame=plot_df, x='time', y='brier_score')
        fig.update_xaxes(title_text='time')
        fig.update_yaxes(title_text='brier score')
        fig.update_layout(title_text='Prediction Error over time', **kwargs)
        if show:
            fig.show()
        return fig

    @property
    def _test_times(self) -> np.ndarray:
        return np.array(sorted(self.y_test[self._time]))

    @property
    def _event_times(self) -> np.ndarray:
        if self.survs is None:
            return np.array([np.nan])
        return self.survs[0].x

    @property
    def _model_enum(self) -> SurvivalModel:
        return SurvivalModel(self.model.__class__.__name__)

    @property
    def _time(self) -> str:
        return self.y_test.dtype.names[1]

    @property
    def proper_times(self) -> np.ndarray:
        event_times, test_times = self._event_times, self._test_times
        min_ = max(min(event_times), min(test_times))
        max_ = min(max(event_times), max(test_times))
        return np.array(sorted(t for t in set(event_times).union(set(test_times)) if min_ <= t < max_))
