import spacy
from nltk.util import skipgrams
from nltk import ngrams, word_tokenize
from collections import Counter, defaultdict
from typing import Union, List, Dict, Tuple, Callable
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class FeaturesClass(object):
    """
    Base class for feature generation. Contains methods that aren't features specific.
    """
    def __init__(self, methods: List[str] = [], cleaner: Callable = None, **kwargs):
        # Initialise variables
        self.args       = kwargs
        self.method     = methods
        self.method_map = {}
        self.features   = {}

        # Initialise imports
        self.tagger  = spacy.load('en')
        self.cleaner = cleaner
        self.sent    = SentimentIntensityAnalyzer()

    @property
    def methods(self):
        return self.method_map

    @methods.setter
    def methods(self, methods):
        for m in methods:
            try:
                self.method_map.update({m: getattr(self, m)})
            except AttributeError as e:
                print("Method {0} not found".format(m))
                raise(e)

    @property
    def doc(self):
        return self.document

    @doc.setter
    def doc(self, document):
        self.document             = document
        self.tokens, self.stopped = self.cleaner(document)

class LinguisticFeatures(FeaturesClass):

    def unigrams(self) -> List[str]:
        """ Returns unigrams after removal of stopwords"""
        return self.tokens

    def token_ngrams(self, **kwargs) -> List[str]:
        return ["_".join(toks) for toks in ngrams(self.tokens, kwargs['ngrams'])]

    def skip_grams(self, **kwargs) -> List[str]:
        return ["_".join(item) for item in skipgrams(self.tokens, kwargs['ngrams'], kwargs['skip_size'])]

    def char_ngrams(self, **kwargs) -> List[str]:
        return ["_".join(toks) for toks in ngrams(" ".join(self.tokens), kwargs['ngrams'])]
