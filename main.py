import numpy as np
import pandas as pd
from dash import Dash
from dash_bootstrap_components.themes import BOOTSTRAP
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
    brca = pd.read_csv(data_path + 'brca.csv', index_col=0)
    X = brca.loc[:, 'age':'FLT1']
    X = pd.get_dummies(X)
    y = np.array(list(zip(brca.status.astype(bool), brca.time)), dtype=[('status', '?'), ('time', '<f8')])

    stage = X.loc[:, "stage"].astype(object).values[:, np.newaxis]
    stage_num = OrdinalEncoder(categories=[[1, 2, 3]]).fit_transform(stage)
    X['stage'] = stage_num

    random_state = 2022

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state,
                                                        stratify=brca.status)

    rsf = RandomSurvivalForest(n_estimators=100,
                               min_samples_split=10,
                               min_samples_leaf=15,
                               max_features="sqrt",
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
