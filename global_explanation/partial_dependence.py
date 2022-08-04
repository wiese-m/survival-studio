import matplotlib.pyplot as plt
import pandas as pd


class PartialDependence:
    def __init__(self, model, X: pd.DataFrame) -> None:
        self.model = model
        self.X = X.copy()

    @property
    def mean_prediction(self) -> float:
        return self.model.predict(self.X).mean()

    def fit(self, feature: str, value) -> None:
        self.X[feature] = value

    def plot(self, X: pd.DataFrame, feature: str) -> None:
        fig, ax = plt.subplots()
        for value in X[feature].unique():
            self.fit(feature, value)
            ax.scatter(value, self.mean_prediction, c='blue')
        ax.set_xlabel(feature)
        ax.set_ylabel('avg risk score')
        fig.suptitle('Partial Dependence')
        plt.show()
