"""Creates features and handles feature generation."""
import spacy
from nltk.util import skipgrams
from nltk import ngrams
from collections import Counter
from typing import List, Callable
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class LinguisticFeatures(object):
    """Linguistic feature generation class."""

    def __init__(self, methods: List[str] = [], cleaner: Callable = None,
                 **kwargs):
        """Set initialisations so that loading only happens once."""
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
        """Handle method mapping."""
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
        """Document handler."""
        return self.document

    @doc.setter
    def doc(self, document):
        self.document             = document
        self.tokens, self.stopped = self.cleaner(document)

    def generate(self):
        """Generate features, where each item is a callable function."""
        if self.method_map == {}:
            self.str_to_method = self.methods

        for m_str in self.method_map:
            self.features.update(
                Counter(self.method_map[m_str](**self.kwargs)))
        return self.features

    def unigrams(self) -> List[str]:
        """Return unigrams after removal of stopwords."""
        return self.tokens

    def token_ngrams(self, **kwargs) -> List[str]:
        """Generate list of token n-grams, n given in kwargs['ngrams'].

        :param kwargs: Keyword Args (must contain 'ngrams').
        :return: list[str]: Multi-token tokens joined by _.
        e.g.: Outstanding blossom -> Outstanding_blossom
        """
        return ["_".join(tok) for tok in ngrams(self.tokens, kwargs['ngrams'])]

    def skip_grams(self, **kwargs) -> List[str]:
        """Generate list of skip-grams.

        :param kwargs: Keyword Args (must contain 'ngrams' and 'skip_size').
        :return: list[str]: Multi-token tokens joined by _.
        """
        return ["_".join(item) for item in skipgrams(self.tokens,
                kwargs['ngrams'], kwargs['skip_size'])]

    def char_ngrams(self, **kwargs) -> List[str]:
        """Generate list of character n-grams.

        :param kwargs: Keyword Args (must contain 'char-ngrams').
        :return: list[str]: Multi-token tokens joined by _.
        """
        return ["_".join(toks) for toks in ngrams(" ".join(self.tokens), kwargs['ngrams'])]
