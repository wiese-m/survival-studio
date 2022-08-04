import dash_loading_spinners as dls
import numpy as np
from dash import Dash, html, dcc, Input, Output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sksurv.datasets import load_gbsg2
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder

from explainer import SurvExplainer
from utils import generate_table, choose_random_feature


def setup_explainer() -> SurvExplainer:
    X, y = load_gbsg2()

    grade_str = X.loc[:, "tgrade"].astype(object).values[:, np.newaxis]
    grade_num = OrdinalEncoder(categories=[["I", "II", "III"]]).fit_transform(grade_str)

    X_no_grade = X.drop("tgrade", axis=1)
    Xt = OneHotEncoder().fit_transform(X_no_grade)
    Xt.loc[:, "tgrade"] = grade_num

    random_state = 20

    X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.25, random_state=random_state)

    rsf = RandomSurvivalForest(n_estimators=1000,
                               min_samples_split=10,
                               min_samples_leaf=15,
                               max_features="sqrt",
                               n_jobs=-1,
                               random_state=random_state)
    rsf.fit(X_train, y_train)

    return SurvExplainer(rsf, X_test, y_test)


def main() -> None:
    explainer = setup_explainer()

    app = Dash(__name__)

    app.layout = html.Div(children=[
        html.H1(children='survivalStudio'),
        html.H4(f'observation id = {explainer.new_observation.index[0]}'),
        generate_table(explainer.new_observation),
        dcc.Dropdown(
            explainer.X.columns,
            choose_random_feature(explainer),
            id='feature-name'
        ),
        dls.Hash(dcc.Graph(id='cpp'))
    ])

    @app.callback(
        Output('cpp', 'figure'),
        Input('feature-name', 'value'))
    def update_cpp_graph(feature):
        return explainer.cp_profile(str(feature)).plot()

    app.run_server(debug=True)


if __name__ == '__main__':
    main()
