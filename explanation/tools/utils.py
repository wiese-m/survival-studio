import pandas as pd


def make_single_observation_by_id(df: pd.DataFrame, id_: int) -> pd.DataFrame:
    return pd.DataFrame(df.loc[id_]).T
