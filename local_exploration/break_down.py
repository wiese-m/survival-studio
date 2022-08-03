import matplotlib.pyplot as plt
import pandas as pd


class BreakDown:
    def __init__(self, model, X: pd.DataFrame, new_observation: pd.DataFrame) -> None:
        self.model = model
        self.X = X
        self.new_observation = new_observation
        self.mean_prediction = self._get_mean_prediction()
        self.results = self._get_results()

    def _get_mean_prediction(self) -> float:
        return self.model.predict(self.X).mean()

    def _get_expected_value_for_features(self, features: list[str]) -> float:
        X = self.X.copy()
        X[features] = self.new_observation[features].squeeze()
        return self.model.predict(X).mean()

    def _get_score_for_feature(self, feature: str) -> float:
        return self._get_expected_value_for_features([feature]) - self.mean_prediction

    def _get_sorted_abs_scores(self) -> dict[str, float]:
        scores = {feature: abs(self._get_score_for_feature(feature)) for feature in self.X.columns}
        return dict(sorted(scores.items(), key=lambda i: i[1], reverse=True))

    def _get_scores_ordering(self) -> list[str]:
        sorted_scores = self._get_sorted_abs_scores()
        return list(sorted_scores.keys())

    def _get_results(self) -> pd.DataFrame:
        ordering = self._get_scores_ordering()
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

    def plot(self) -> None:
        fig, ax = plt.subplots()
        ax.step(self.results.cumulative[::-1], self.results.variable_name[::-1], where='post')
        ax.scatter(self.results.cumulative[::-1], self.results.variable_name[::-1])
        ax.set_xlabel('risk score')
        fig.suptitle('Break Down')
        plt.show()
