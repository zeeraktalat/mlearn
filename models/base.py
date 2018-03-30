from typing import Generator
import joblib

class baseModel(object):
    def __init__(self, data: Generator, **kwargs):
        """Class containing implementations and API calls to embedding models.
        :param data: dataset loaded as a generator."""

        self.dataset    = data
        self.model_path = MODELPATH
        self.model      = None
        self.kwargs     = kwargs

    @property
    def model_handler(self):
        return self.model

    @model_handler.setter
    def model_handler(self, model):
        self.model = model

    def save_model(self, model, model_name):
        """Save model to directory.
        :param model: Model to dump
        :param model_name: Filename for model
        """

        if model_name == 'gensim-wv2':
            model_name = 'gensim_w2v.model'

        joblib.dump(model, model_name + self.model_path)

    def load_model(self, model: str):
        if model == 'gensim-w2v':
            model_name = 'gensim_w2v.model'

        self.model = joblib.load(model_name + self.model_path)

        return self.model
