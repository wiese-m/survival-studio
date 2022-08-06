import numpy as np
import pandas as pd
from sksurv.metrics import (
    concordance_index_ipcw,
    brier_score,
    integrated_brier_score
)


class ModelPerformance:
    def __init__(self, model, X_test: pd.DataFrame) -> None:
        self.model = model
        self.X_test = X_test
        self.survs = model.predict_survival_function(X_test)

    def harrell_cindex(self, X: pd.DataFrame, y: np.ndarray) -> float:
        return self.model.score(X, y)

    def uno_cindex(self, y_train: np.ndarray, y_test: np.ndarray, tau: float = None, tied_tol: float = 1e-8) -> float:
        cindex, concordant, discordant, tied_risk, tied_time = \
            concordance_index_ipcw(y_train, y_test, self.model.predict(self.X_test), tau, tied_tol)
        return cindex

    def brier_score(self, y_train: np.ndarray, y_test: np.ndarray, time) -> float:
        preds = [surv_func(time) for surv_func in self.survs]
        times, score = brier_score(y_train, y_test, preds, time)
        return score[0]

    def integrated_brier_score(self, y_train: np.ndarray, y_test: np.ndarray, times) -> float:
        preds = np.asarray([[surv_func(t) for t in times] for surv_func in self.survs])
        return integrated_brier_score(y_train, y_test, preds, times)
