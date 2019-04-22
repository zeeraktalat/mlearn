"""A module to create a layer over Sklearn methods."""

from typing import Generator
from z_api.base.modelBase import baseModel


class sklearnClassifier(baseModel):
    """A class to implement Sklearn classifiers."""
    def __init__(self, model, data: Generator, labels: list, **kwargs):
        self.data = data
        self.labels = labels
        self.model = model
        self.kwargs = kwargs
        super(sklearnClassifier, self).__init__(data, labels)

    @property
    def model_handler(self):
        """Get and set model to be used."""
        return self.model

    @model_handler.setter
    def model_handler(self, model):
        self.model = model

    def fit(self, X, y) -> None:
        """Fit model.

        :param X: Training data, vectorised.
        :param y: Labels for training set.
        """
        self.model.fit(X, y)

    def _ensure_attributes(self, key, value) -> None:
        """Make sure all attributes that are commonly used are available.
        :param key: Key for the attribute to be stored ind.
        :param value: Value for the key.
        """
        try:
            assert key in self.model.__dict__
        except AssertionError as e:
            setattr(self.model, key, value)

    def predict(self, test):
        """Predict using self.model.

        :param test: Data to run prediction on.
        :return pred: Return predictions on test set.
        """
        pred = self.model.predict(test)
        return pred

    def fit_predict(self, X, y, test):
        """Fit model and predict. Equivalent to fit followed by predict.

        :param X: Training data, vectorised.
        :param y: Labels for training set.
        :param test: Test data.
        """
        self.fit(X, y)
        return self.predict(test)

    def cross_validate(self, folds: int = 10):
        """Cross validate using a given number of folds.

        :param folds: int: The number of folds to use.
        """
        raise NotImplementedError

    def reductions_fairness(self, criteria: str = 'DP'):
        """Run Agarwal et al. a reductions approach to fairness.
        :param criteria: Fairness criteria
        """
        raise NotImplementedError
