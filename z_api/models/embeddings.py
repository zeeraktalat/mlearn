"""Module for implementing methods for generating embeddings."""


# from mlapi.utils.readers import MongoDB
from gensim.models.word2vec import Word2Vec
# from typing import Generator
import base

# TODO Add logging


class Embeddings(base.baseModel):
    """Implements various embedding methods. Inherents from base.baseModel."""

    def word2vec(self):
        """Trains Word2Vec model and returns it.

        :return: Trained Word2Vec model
        """
        model = Word2Vec(self.dataset,
                         size = self.kwarg['size'],
                         window = self.kwargs['window'],
                         min_count = self.kwargs['min_count']
                         )
        return model

    def load_word2vec(self, filepath):
        """Load Word2Vec model.

        :param filepath: Path to model
        :ruturn: Loaded word2vec model
        """
        return Word2Vec.load(filepath)

    def save_word2vec(self, model, filepath):
        """Save Word2Vec model.

        :param model: Model to save
        :param filepath: Path to model
        :ruturn: Loaded word2vec model
        """
        return model.save(filepath)
