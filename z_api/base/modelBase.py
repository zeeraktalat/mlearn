"""A module to have basic model operations such as loading/saving, data set splitting, etc."""

import re
import os
import joblib
from gensim.models.word2vec import Word2Vec
from mlapi.utils.logger import initialise_loggers

MODELPATH = os.environ['MODELPATH']


class baseModel(object):
    """A class to contain basic things all models share regardless of which model it is."""

    def __init__(self, **kwargs):
        """Class containing implementations and API calls to embedding models.
        :param kwargs: Must contain training and test set.
        """
        # self.dataset    = data
        # self.labels     = labels
        self.model_path = MODELPATH
        self.kwargs     = kwargs
        self.log        = initialise_loggers('ModelFactory', 'logs/api.log')

    @property
    def training_set(self):
        """Get and set training set."""
        return self.X

    @training_set.setter
    def training_set(self, data):
        self.X = data

    @property
    def dev_set(self):
        """Get and set dev set."""
        return self.dev

    @dev_set.setter
    def dev_set(self, data):
        self.dev = data

    @property
    def test_set(self):
        """Get and set test set."""
        return self.dev

    @test_set.setter
    def test_set(self, data):
        self.test = data

    def save_model(self, model, model_name: str, library: str = 'sklearn'):
        """Save model to directory.

        :param model: str: Model to dump
        :param model_name: str: Filename for model
        :param library: str: Library used to train model
        """
        model_name = "{0}.model".format(re.sub('-', '_', model_name))
        model_path = "{0}/{1}/{2}".format(self.model_path, library, model_name)

        if library == 'sklearn':
            joblib.dump(model, model_path)
        elif library == 'gensim':
            model.save(model_path)
        elif library == 'pytorch':
            # TODO Save pytorch model
            pass
        self.log.info('Model stored in {0}'.format(model_path))

    def load_model(self, model: str, library: str, **kwargs):
        """Load model from directory.

        :param model: str: Model to load
        :param library: str: Library to load.
        :return self.model: Return loaded model
        """
        model_name = "{0}.model".format(re.sub('-', '_', model))
        model_path = "{0}/{1}/{2}".format(self.model_path, library, model_name)

        if library == 'sklearn':
            self.model = joblib.load(model_path)
        elif library == 'gensim':
            if model == 'word2vec':
                Word2Vec.load(self.model_path)
        elif library == 'pytorch':
            # TODO Save pytorch model
            pass
        return self.model
