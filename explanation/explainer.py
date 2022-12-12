import random

import numpy as np
import pandas as pd
import shap

from explanation.global_explanation.feature_importance import FeatureImportance
from explanation.global_explanation.model_performance import ModelPerformance
from explanation.global_explanation.partial_dependence import PartialDependence
from explanation.local_explanation.break_down import BreakDown
from explanation.local_explanation.ceteris_paribus import CeterisParibus
from explanation.tools.visualizer import Visualizer


# Survival analysis explanations handler
class SurvExplainer:
    def __init__(self, model, X: pd.DataFrame, y: np.ndarray, model_name: str = None) -> None:
        self.model = model
        self.model_name = model_name if model_name is not None else self._model_name
        self.X = X
        self.y = y
        self.new_observation = self.choose_random_observation()
        self.visualizer = Visualizer(self.model, self.X, self.new_observation)
        self.model_performance = ModelPerformance(self.model, self.X, self.y)

    # Compute prediction for given data
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    # Choose random observation from dataset
    def choose_random_observation(self) -> pd.DataFrame:
        return pd.DataFrame(self.X.loc[random.choice(self.X.index)]).T

    # Make Ceteris Paribus Profile instance for given observation and feature
    def cp_profile(self, feature: str, new_observation: pd.DataFrame = None) -> CeterisParibus:
        new_observation = self.new_observation if new_observation is None else new_observation
        return CeterisParibus(self.model, self.X, new_observation, feature)

    # Make (interaction) Break Down profile instance for given observation
    def bd_profile(self, allow_interactions: bool = False, new_observation: pd.DataFrame = None) -> BreakDown:
        new_observation = self.new_observation if new_observation is None else new_observation
        return BreakDown(self.model, self.X, new_observation, allow_interactions)

    # Make Partial Dependence profile instance for given feature
    def pd_profile(self, feature: str) -> PartialDependence:
        return PartialDependence(self.model, self.X, feature)

    # Compute SHAP values explanations (currently not used in dashboard)
    def shap_values(self, n: int) -> shap.Explanation:
        X = self.X.sample(n)
        shap_explainer = shap.Explainer(self.model.predict, X, feature_names=X.columns)
        return shap_explainer(X)

    # Make Permutation Feature Importance instance
    def feature_importance(self, n_iter: int = 5, random_state: int = None) -> FeatureImportance:
        return FeatureImportance(self.model, self.X, self.y, n_iter, random_state)

    @property
    # Get model name from class name
    def _model_name(self) -> str:
        return self.model.__class__.__name__
