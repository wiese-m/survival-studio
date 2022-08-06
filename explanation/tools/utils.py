import random

import pandas as pd
from dash import html

from explanation.explainer import SurvExplainer


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


def choose_random_feature(explainer: SurvExplainer) -> str:
    return random.choice(explainer.X.columns)


def make_single_observation_by_id(df: pd.DataFrame, id_: int) -> pd.DataFrame:
    return pd.DataFrame(df.loc[id_]).T
