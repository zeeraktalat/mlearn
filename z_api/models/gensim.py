"""Module for implementing methods for generating embeddings."""


# from mlapi.utils.readers import MongoDB
from gensim.models.word2vec import Word2Vec
# from typing import Generator
import api_z.models.base as base

# TODO Add logging
# TODO Add everything else


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
