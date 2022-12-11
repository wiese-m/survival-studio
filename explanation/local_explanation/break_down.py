from itertools import combinations, compress

import pandas as pd
import plotly.graph_objects as go


class BreakDown:
    def __init__(self, model, X: pd.DataFrame, new_observation: pd.DataFrame, allow_interactions: bool) -> None:
        self._allow_interactions = allow_interactions
        self.model = model
        self.X = X
        self.new_observation = new_observation
        self.mean_prediction = self._get_mean_prediction()
        self._single_scores = self._get_single_scores()
        self._pairwise_scores = self._get_pairwise_scores() if allow_interactions else {}
        self.result = self._get_results()

    def plot(self, show: bool = False, **kwargs) -> go.Figure:
        fig = go.Figure()
        measure = ['relative'] * self.result.shape[0]
        measure[0] = 'absolute'
        measure[-1] = 'total'
        fig = fig.add_waterfall(
            x=self.result.vimp,
            y=self.result.variable_name,
            orientation='h',
            measure=measure
        )
        fig.update_xaxes(title_text='risk score')
        fig.update_yaxes(title_text='')
        title = 'Break Down' if not self._allow_interactions else 'interaction Break Down'
        fig.update_layout(title_text=title, **kwargs)
        if show:
            fig.show()
        return fig

    def _get_mean_prediction(self) -> float:
        return self.model.predict(self.X).mean()

    def _get_expected_value_for_features(self, features: list[str]) -> float:
        X = self.X.copy()
        X[features] = self.new_observation[features].squeeze()
        return self.model.predict(X).mean()

    def _get_single_scores(self) -> dict[str, float]:
        return {feature: self._get_delta({feature}, {}) for feature in self.X.columns}

    def _get_delta(self, L: {str}, J: {str}) -> float:
        assert all([x in self.X.columns for x in L])
        assert all([x in self.X.columns for x in J])
        assert L.isdisjoint(J)
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

    def _get_sorted_proper_scores(self, reverse: bool = True) -> dict[str, float]:
        all_scores = self._single_scores | self._pairwise_scores
        all_scores = {feature: score for feature, score in all_scores.items() if feature in self._get_proper_features()}
        return self._sorted_dict_by_abs_values(all_scores, reverse)

    @staticmethod
    def _sorted_dict_by_abs_values(d: dict, reverse: bool) -> dict:
        return dict(sorted(d.items(), key=lambda i: abs(i[1]), reverse=reverse))

    def _get_results(self) -> pd.DataFrame:
        vimp_df = pd.DataFrame([self._get_vimp()]).T[::-1].reset_index() \
            .rename(columns={'index': 'variable_name', 0: 'vimp'})
        vimp_df.loc[-1] = ['intercept', self.mean_prediction]
        vimp_df.loc[999] = ['prediction', 0]
        vimp_df = vimp_df.sort_index().reset_index(drop=True)
        vimp_df['break_down'] = vimp_df.vimp.cumsum()
        return vimp_df

    def _get_vimp(self) -> dict[str, float]:
        vimp = {}
        previous_features = []
        features = self._get_proper_features_for_vimp(self._get_sorted_proper_scores())
        for feature in features[::-1]:
            if ':' not in feature:
                vimp[feature] = self._get_delta({feature}, set(previous_features))
                previous_features.append(feature)
            else:  # Interaction A:B
                features_ = feature.split(':')
                vimp[feature] = self._get_delta(set(features_), set(previous_features))
                previous_features.append(features_)
            previous_features = self._flatten(previous_features)
        return vimp

    def _flatten(self, list_: list) -> list:
        result = []
        for x in list_:
            if isinstance(x, list):
                result.extend(self._flatten(x))
            else:
                result.append(x)
        return result

    def _get_proper_features_for_vimp(self, sorted_proper_scores: dict[str, float]) -> list[str]:
        checks = []
        features = list(sorted_proper_scores.keys())
        for i, feature in enumerate(features):
            checks.append([f in features[i - 1] for f in feature.split(':')])
        checks = [(i, self._negate(check)) for i, check in enumerate(checks) if any(check)]
        for check in checks:
            features[check[0]] = list(compress(features[check[0]].split(':'), check[1]))[0]
        return features

    @staticmethod
    def _negate(boolean_list: list[bool]) -> list[bool]:
        return [not i for i in boolean_list]
