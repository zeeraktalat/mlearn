from mlapi.utils.readers import MongoDB
from gensim.models.word2vec import Word2Vec
from typing import Generator
import base


class Embeddings(base.baseModel):
    """Implements various embedding methods. Inherents from base.baseModel"""

    def word2vec(self):
        """Trains Word2Vec model and returns it
        :return: Trained Word2Vec model
        """
        model = Word2Vec(self.dataset,
                         size = self.kwarg['size'],
                         window = self.kwargs['window'],
                         min_count = self.kwargs['min_count']
                         )
        return model

    def load_word2vec(self, filepath):
        return Word2Vec.load(filepath)

    def save_word2vec(self, model, filepath):
        return model.save(filepath)
