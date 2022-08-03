import matplotlib.pyplot as plt
import pandas as pd


class CeterisParibus:
    def __init__(self, model, new_observation: pd.DataFrame) -> None:
        self.model = model
        self.new_observation = new_observation.copy()

    def fit(self, feature: str, value) -> None:
        self.new_observation[feature] = value

    @property
    def new_prediction(self) -> float:
        return self.model.predict(self.new_observation)

    def plot(self, X: pd.DataFrame, feature: str) -> None:
        fig, ax = plt.subplots()
        ax.scatter(self.new_observation[feature], self.new_prediction, c='red', zorder=2)
        for value in X[feature].unique():
            self.fit(feature, value)
            ax.scatter(value, self.new_prediction, c='blue', zorder=1)
        ax.set_xlabel(feature)
        ax.set_ylabel('risk score')
        fig.suptitle('Ceteris Paribus')
        plt.show()
