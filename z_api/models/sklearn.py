"""A module to create a layer over Sklearn methods."""

from z_api.base.modelBase import baseModel
from typing import Generator


class sklearnClassifer(baseModel):
    """A class to implement Sklearn classifiers."""
    def __init__(self, model, data: Generator, labels: list, **kwargs):
        self.data = data
        self.labels = labels
        self.model = model
        self.kwargs = kwargs
        super(sklearnClassifer, self).__init__(data, labels)

    @property
    def model_handler(self):
        """Get and set model to be used."""
        return self.model

    @model_handler.setter
    def model_handler(self, model):
        self.model = model

    def fit(self, X, y):
        """Fit model.

        :param X: Training data, vectorised.
        :param y: Labels for training set.
        """
        self.model.fit(X, y)

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
