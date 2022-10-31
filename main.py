import numpy as np
import pandas as pd
from dash import Dash
from dash_bootstrap_components.themes import BOOTSTRAP
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sksurv.datasets import load_gbsg2
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder

from explanation.explainer import SurvExplainer
from src.components.layout import create_layout


def setup_gbsg2_explainer() -> SurvExplainer:
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


def setup_brca_explainer(data_path: str = 'data/files/') -> SurvExplainer:
    brca = pd.read_csv(data_path + 'brca-v2.csv', index_col=0)

    random_state = 2022

    X_smote, y_smote = brca.drop(columns=['status']), brca.status
    X_smote['pos_lymphnodes'] = X_smote.pos_lymphnodes.astype(np.int64)
    sm = SMOTENC(random_state=random_state, categorical_features=[4, 5, 6, 7])
    X_res, y_res = sm.fit_resample(X_smote, y_smote)
    brca_res = X_res.copy()
    brca_res['status'] = y_res

    X = brca_res.loc[:, 'age':'BRCA2']
    stage = X.loc[:, "stage"].astype(object).values[:, np.newaxis]
    stage_num = OrdinalEncoder(categories=[['I', 'II', 'III']]).fit_transform(stage)
    X_no_stage = X.drop(columns='stage')
    X = pd.get_dummies(X_no_stage, drop_first=True)
    y = np.array(list(zip(brca_res.status.astype(bool), brca_res.time)), dtype=[('status', '?'), ('time', '<f8')])
    X['stage'] = stage_num

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=random_state)

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


def main() -> None:
    explainer = setup_brca_explainer()
    app = Dash(external_stylesheets=[BOOTSTRAP])
    app.title = "Survival Studio"
    app.layout = create_layout(app, explainer)
    app.run()


if __name__ == "__main__":
    main()
