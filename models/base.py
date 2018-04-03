"""A module to have basic model operations such as loading/saving, prediction, etc."""

import re
import os
import joblib
from typing import Generator
from gensim.models.word2vec import Word2Vec
from mlapi.utils.logger import initialise_loggers

MODELPATH = os.environ['MODELPATH']


class baseModel(object):
    """A class to contain basic things all models share regardless of which model it is."""

    def __init__(self, data: Generator, **kwargs):
        """Class containing implementations and API calls to embedding models.

        :param data: dataset loaded as a generator.
        """
        self.dataset    = data
        self.model_path = MODELPATH
        self.model      = None
        self.kwargs     = kwargs
        self.log        = initialise_loggers('ModelFactory', 'logs/api.log')

    @property
    def model_handler(self):
        """Get and set model to be used."""
        return self.model

    @model_handler.setter
    def model_handler(self, model):
        self.model = model

    def save_model(self, model, model_name: str):
        """Save model to directory.

        :param model: Model to dump
        :param model_name: Filename for model
        """
        if 'gensim-w2v' in model_name:
            model_name = re.sub('-', '_', model_name)
            self._save_gensim_word2vec(model, model_name + '.model')
            self.log.info('Model stored in {0}'.format(self.model_path + model_name + '.model'))

            return None

        elif 'LogisticRegression' in model_name:
            model_name = model_name + '.model'

        joblib.dump(model, model_name + self.model_path)

    def load_model(self, model: str):
        """Load model from directory.

        :param model: str: Model to load
        :return self.model: Return loaded model
        """
        model_name = ""

        if 'gensim-w2v' in model_name:
            model_name = 'gensim_w2v.model'

        self.model = joblib.load(model_name + self.model_path)

        return self.model

    def _load_gensim_word2vec(self, size):
        return Word2Vec.load(self.model_path + 'gensim_w2v.{0}.model'.format(size))

    def _save_gensim_word2vec(self, model, model_name):
        return model.save(self.model_path + model_name)

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
