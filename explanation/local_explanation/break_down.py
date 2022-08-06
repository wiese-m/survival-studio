from itertools import combinations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class BreakDown:
    def __init__(self, model, X: pd.DataFrame, new_observation: pd.DataFrame, allow_interactions: bool) -> None:
        self.model = model
        self.X = X
        self.new_observation = new_observation
        self.mean_prediction = self._get_mean_prediction()
        self._single_scores = self._get_single_scores()
        self._pairwise_scores = self._get_pairwise_scores() if allow_interactions else {}
        self.result = self._get_results()  # todo: need to be changed for interaction features like A:B

    def _get_mean_prediction(self) -> float:
        return self.model.predict(self.X).mean()

    def _get_expected_value_for_features(self, features: list[str]) -> float:
        X = self.X.copy()
        X[features] = self.new_observation[features].squeeze()
        return self.model.predict(X).mean()

    def _get_single_scores(self) -> dict[str, float]:
        return {feature: self._get_delta({feature}, {}) for feature in self.X.columns}

    def _get_results(self) -> pd.DataFrame:
        ordering = list(self._get_sorted_proper_scores().keys())
        features = []
        values = []
        for feature in ordering:
            features.append(feature)
            values.append(self._get_expected_value_for_features(features))
        results = pd.DataFrame({'variable_name': ordering,
                                'variable_value': self.new_observation[ordering].values[0],
                                'cumulative': values})
        intercept = pd.DataFrame([['intercept', self.mean_prediction, self.mean_prediction]], columns=results.columns)
        new_prediction = self.model.predict(self.new_observation)[0]
        prediction = pd.DataFrame([['prediction', new_prediction, new_prediction]], columns=results.columns)
        results = pd.concat([intercept, results, prediction]).reset_index(drop=True)
        results['contribution'] = results.cumulative.diff()
        return results

    def plot(self, show: bool = False) -> go.Figure:
        # todo: change to waterfall plot
        fig = px.line(x=self.result.cumulative[::-1], y=self.result.variable_name[::-1], line_shape='hv')
        fig.update_traces(mode='lines+markers')
        fig.update_xaxes(title_text='risk score')
        fig.update_yaxes(title_text='')
        fig.update_layout(title_text='Break Down', width=400, height=300)
        if show:
            fig.show()
        return fig

    def _get_delta(self, L: {str}, J: {str}) -> float:
        X_copy1, X_copy2 = self.X.copy(), self.X.copy()
        X_copy1[list(L.union(J))] = self.new_observation[list(L.union(J))].squeeze()
        if J:
            X_copy2[list(J)] = self.new_observation[list(J)].squeeze()
        return self.model.predict(X_copy1).mean() - self.model.predict(X_copy2).mean()

    def _get_interaction_delta(self, i: str, j: str) -> float:
        return self._get_delta({i, j}, {}) - self._get_delta({i}, {}) - self._get_delta({j}, {})

    def _get_pairwise_scores(self) -> dict[str, float]:
        return {f'{i}:{j}': self._get_interaction_delta(i, j) for i, j in combinations(self.X.columns, 2)}

    def _get_signif_interactions(self) -> list[str]:
        signif_interactions = []
        for feature, score in self._pairwise_scores.items():
            single_features = feature.split(':')
            if abs(score) > abs(self._single_scores[single_features[0]]) and \
                    abs(score) > abs(self._single_scores[single_features[1]]):
                signif_interactions.append(feature)
        return signif_interactions

    def _get_proper_features(self) -> list[str]:
        signif_interactions = self._get_signif_interactions()
        to_remove = [feature for feature in self._single_scores if any(feature in i for i in signif_interactions)]
        return [f for f in list(self._single_scores.keys()) + signif_interactions if f not in to_remove]

    def _get_sorted_proper_scores(self) -> dict[str, float]:
        all_scores = self._single_scores | self._pairwise_scores
        all_scores = {feature: score for feature, score in all_scores.items() if feature in self._get_proper_features()}
        return self._sorted_dict_by_abs_values(all_scores)

    @staticmethod
    def _sorted_dict_by_abs_values(d: dict) -> dict:
        return dict(sorted(d.items(), key=lambda i: abs(i[1]), reverse=True))
