from collections import defaultdict

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sksurv.datasets import load_gbsg2
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder

from explanation.explainer import SurvExplainer


def setup_rsf_gbsg2_explainer(random_state: int = 2022) -> SurvExplainer:
    X, y = load_gbsg2()

    grade_str = X.loc[:, "tgrade"].astype(object).values[:, np.newaxis]
    grade_num = OrdinalEncoder(categories=[["I", "II", "III"]]).fit_transform(grade_str)

    X_no_grade = X.drop("tgrade", axis=1)
    Xt = OneHotEncoder().fit_transform(X_no_grade)
    Xt.loc[:, "tgrade"] = grade_num

    X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.25, random_state=random_state)

    rsf = RandomSurvivalForest(n_estimators=500,
                               min_samples_split=10,
                               min_samples_leaf=15,
                               max_features="sqrt",
                               n_jobs=-1,
                               random_state=random_state)
    rsf.fit(X_train, y_train)

    return SurvExplainer(rsf, X_test, y_test)


def _prepare_brca_data(data_path: str = 'data/', balanced: bool = False, random_state: int = 2022) -> tuple:
    brca = pd.read_csv(data_path + 'brca-v2.csv', index_col=0)

    brca['pos_lymphnodes'] = brca.pos_lymphnodes.astype(np.int64)
    brca['tumor_weight'] = brca.tumor_weight.astype(np.int64)

    X = brca.drop(columns=['time', 'status'])
    stage = X.loc[:, "stage"].astype(object).values[:, np.newaxis]
    stage_num = OrdinalEncoder(categories=[['I', 'II', 'III']]).fit_transform(stage)
    X_no_stage = X.drop(columns='stage')
    X = pd.get_dummies(X_no_stage, drop_first=True)
    y = np.array(list(zip(brca.status.astype(bool), brca.time)), dtype=[('status', '?'), ('time', '<f8')])
    X['stage'] = stage_num

    # source: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
    corr = spearmanr(X).correlation
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    cluster_ids = hierarchy.fcluster(dist_linkage, 0.6, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=random_state,
                                                        stratify=brca.status)
    X_train = X_train.iloc[:, selected_features]
    X_test = X_test.iloc[:, selected_features]

    # SMOTE training data
    if balanced:
        X_smote, y_smote = X_train.copy(), y_train['status'].copy()
        X_smote['time'] = y_train['time'].astype(np.int64)
        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_smote, y_smote)
        y_train = np.array(list(zip(y_train, X_train.time)), dtype=[('status', '?'), ('time', '<f8')])
        X_train.drop(columns='time', inplace=True)

    return X_train, X_test, y_train, y_test


def setup_rsf_brca_explainer(data_path: str = 'data/',
                             balanced: bool = False,
                             random_state: int = 2022) -> SurvExplainer:
    X_train, X_test, y_train, y_test = _prepare_brca_data(data_path, balanced, random_state)
    rsf = RandomSurvivalForest(n_jobs=-1, random_state=random_state)
    rsf.fit(X_train, y_train)
    return SurvExplainer(rsf, X_test, y_test)


def setup_coxph_brca_explainer(data_path: str = 'data/',
                               balanced: bool = False,
                               random_state: int = 2022) -> SurvExplainer:
    X_train, X_test, y_train, y_test = _prepare_brca_data(data_path, balanced, random_state)
    coxph = CoxPHSurvivalAnalysis()
    coxph.fit(X_train, y_train)
    return SurvExplainer(coxph, X_test, y_test)


def setup_gbm_brca_explainer(loss='coxph',
                             data_path: str = 'data/',
                             balanced: bool = False,
                             random_state: int = 2022) -> SurvExplainer:
    X_train, X_test, y_train, y_test = _prepare_brca_data(data_path, balanced, random_state)
    gbm = GradientBoostingSurvivalAnalysis(loss=loss, random_state=random_state)
    gbm.fit(X_train, y_train)
    return SurvExplainer(gbm, X_test, y_test)
