"""This module creates vectoriser and related methods."""

class Vectorisation(object):
    """Set up the vectoriser."""

    def __init__(self, X, y, vectoriser):
        """Initialise vectoriser."""
        self.vect = vectoriser

    @property
    def vectoriser(self):
        """Vectoriser method."""
        return self.vect

    @vectoriser.setter
    def vectoriser(self, vectoriser):
        self.vect = vectoriser

    def train(self, X):
        """Train the vectoriser.

        :param X: Training data.
        :return: Fit and transformed matrix of training data
        """
        return self.vect.fit_transform(X)

    def test(self, X):
        """Train the vectoriser.

        :param X: Test data.
        :return: Transformed matrix of test data
        """
        return self.vect.transform(X)
