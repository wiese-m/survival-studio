import dash_loading_spinners as dls
import numpy as np
from dash import Dash, html, dcc, Input, Output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sksurv.datasets import load_gbsg2
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder

import utils as ut
from explainer import SurvExplainer


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
    mp = explainer.model_performance()
    # y_time = explainer.y[explainer.y.dtype.names[1]]
    brier_time = explainer.model.event_times_

    app = Dash(__name__)

    app.layout = html.Div(children=[
        html.H1(children='survivalStudio'),
        html.Div(children=[
            dcc.Dropdown(
                explainer.X.index,
                explainer.X.index[0],
                id='new-observation-id'
            )
        ], style={'width': '20%'}),
        html.Div(id='obs-table'),
        html.Div(children=[
            dcc.Dropdown(
                explainer.X.columns,
                ut.choose_random_feature(explainer),
                id='feature-name'
            )
        ], style={'width': '20%'}),
        dls.Hash(dcc.Graph(id='cpp')),
        dls.Hash(dcc.Graph(id='bdp')),
        dls.Hash(dcc.Graph(id='pdp')),
        dls.Hash(dcc.Graph(id='surv')),
        dcc.Graph(figure=explainer.feature_importance().plot()),
        # todo: moze tutaj zamiast tego tez slider?
        html.Div([
            dcc.Input(id='time', type='number', placeholder=f'from {brier_time.min()} to {brier_time.max()}',
                      min=brier_time.min(), max=brier_time.max())
        ]),
        html.H5(id='brier-score'),
        html.Div([
            dcc.RangeSlider(brier_time.min(), brier_time.max(), id='time-slider',
                            tooltip={"placement": "bottom", "always_visible": True})
        ]),
        html.H5(id='integrated-brier-score'),
        html.H6(f'''
            harrell_cindex = {mp.harrell_cindex(explainer.X, explainer.y)}
            uno_cindex = {mp.uno_cindex(explainer.y, explainer.y)}
            brier_score = {mp.brier_score(explainer.y, explainer.y, 500)}
            ibs = {mp.integrated_brier_score(explainer.y, explainer.y, [500, 501])}
        ''')
    ])

    @app.callback(
        Output('integrated-brier-score', 'children'),
        [Input('time-slider', 'value')])
    def update_integrated_brier_score(times):
        ibs = mp.integrated_brier_score(explainer.y, explainer.y, times)
        return f'IBS({times}) = {ibs:.4f}'

    @app.callback(
        Output('brier-score', 'children'),
        Input('time', 'value'))
    def update_brier_score(time):
        bs = mp.brier_score(explainer.y, explainer.y, time)
        return f'BS({time}) = {bs:.4f}'

    @app.callback(
        Output('cpp', 'figure'),
        Input('new-observation-id', 'value'),
        Input('feature-name', 'value'))
    def update_cpp_graph(new_observation_id, feature):
        return explainer \
            .cp_profile(str(feature), ut.make_single_observation_by_id(explainer.X, new_observation_id)) \
            .plot()

    @app.callback(
        Output('bdp', 'figure'),
        Input('new-observation-id', 'value'))
    def update_bdp_graph(new_observation_id):
        return explainer \
            .bd_profile(new_observation=ut.make_single_observation_by_id(explainer.X, new_observation_id)) \
            .plot()

    @app.callback(
        Output('pdp', 'figure'),
        Input('feature-name', 'value'))
    def update_cpp_graph(feature):
        return explainer.pd_profile(str(feature)).plot()

    @app.callback(
        Output('surv', 'figure'),
        Input('new-observation-id', 'value'))
    def update_surv_graph(new_observation_id):
        return explainer \
            .visualizer \
            .plot_surv(ut.make_single_observation_by_id(explainer.X, new_observation_id))

    @app.callback(
        Output('obs-table', 'children'),
        Input('new-observation-id', 'value'))
    def update_obs_table(new_observation_id):
        return ut.generate_table(ut.make_single_observation_by_id(explainer.X, new_observation_id))

    app.run_server(debug=True)


# todo: add histogram with nbins slider
if __name__ == '__main__':
    main()
