import pandas as pd


# Make (1 x p) DataFrame with observation info (values of p features) for given index
def make_single_observation_by_id(df: pd.DataFrame, id_: int) -> pd.DataFrame:
    return pd.DataFrame(df.loc[id_]).T
