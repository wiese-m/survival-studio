import random

import pandas as pd

from explanation.explainer import SurvExplainer


def choose_random_feature(explainer: SurvExplainer) -> str:
    return random.choice(explainer.X.columns)


def make_single_observation_by_id(df: pd.DataFrame, id_: int) -> pd.DataFrame:
    return pd.DataFrame(df.loc[id_]).T
