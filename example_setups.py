import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sksurv.datasets import load_gbsg2
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder

from explanation.explainer import SurvExplainer


def setup_rsf_gbsg2_explainer() -> SurvExplainer:
    X, y = load_gbsg2()

    grade_str = X.loc[:, "tgrade"].astype(object).values[:, np.newaxis]
    grade_num = OrdinalEncoder(categories=[["I", "II", "III"]]).fit_transform(grade_str)

    X_no_grade = X.drop("tgrade", axis=1)
    Xt = OneHotEncoder().fit_transform(X_no_grade)
    Xt.loc[:, "tgrade"] = grade_num

    random_state = 20
    print(Xt)
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.25, random_state=random_state)

    rsf = RandomSurvivalForest(n_estimators=1000,
                               min_samples_split=10,
                               min_samples_leaf=15,
                               max_features="sqrt",
                               n_jobs=-1,
                               random_state=random_state)
    rsf.fit(X_train, y_train)

    return SurvExplainer(rsf, X_test, y_test)


def setup_rsf_brca_explainer(data_path: str = 'data/', balanced: bool = True) -> SurvExplainer:
    brca = pd.read_csv(data_path + 'brca-v2.csv', index_col=0)

    random_state = 2022

    X = brca.loc[:, 'age':'BRCA2']
    stage = X.loc[:, "stage"].astype(object).values[:, np.newaxis]
    stage_num = OrdinalEncoder(categories=[['I', 'II', 'III']]).fit_transform(stage)
    X_no_stage = X.drop(columns='stage')
    X = pd.get_dummies(X_no_stage, drop_first=True)
    y = np.array(list(zip(brca.status.astype(bool), brca.time)), dtype=[('status', '?'), ('time', '<f8')])
    X['stage'] = stage_num

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=random_state)

    # SMOTE training data
    if balanced:
        X_smote, y_smote = X_train.copy(), y_train['status'].copy()
        X_smote['time'] = y_train['time'].astype(np.int64)
        X_smote['pos_lymphnodes'] = X_smote.pos_lymphnodes.astype(np.int64)
        sm = SMOTENC(random_state=random_state, categorical_features=[10, 11, 12, 13])
        X_train, y_train = sm.fit_resample(X_smote, y_smote)
        y_train = np.array(list(zip(y_train, X_train.time)), dtype=[('status', '?'), ('time', '<f8')])
        X_train.drop(columns='time', inplace=True)

    rsf = RandomSurvivalForest(n_estimators=250,
                               max_depth=4,
                               min_samples_split=10,
                               min_samples_leaf=15,
                               max_features="sqrt",
                               oob_score=True,
                               n_jobs=-1,
                               random_state=random_state)
    rsf.fit(X_train, y_train)

    return SurvExplainer(rsf, X_test, y_test)
