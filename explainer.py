import random

import numpy as np
import pandas as pd

from local_explanation.ceteris_paribus import CeterisParibus
from local_explanation.break_down import BreakDown


class SurvExplainer:
    def __init__(self, model, X: pd.DataFrame, y: np.ndarray) -> None:
        self.model = model
        self.X = X
        self.y = y
        self.new_observation = self.choose_random_observation()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def choose_random_observation(self) -> pd.DataFrame:
        return pd.DataFrame(self.X.loc[random.choice(self.X.index)]).T

    def cp_profile(self, feature: str, new_observation: pd.DataFrame = None) -> CeterisParibus:
        new_observation = self.new_observation if new_observation is None else new_observation
        return CeterisParibus(self.model, self.X, new_observation, feature)

    def bd_profile(self, allow_interactions: bool = False, new_observation: pd.DataFrame = None) -> BreakDown:
        new_observation = self.new_observation if new_observation is None else new_observation
        return BreakDown(self.model, self.X, new_observation, allow_interactions)
