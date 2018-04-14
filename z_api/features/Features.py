"""Creates features and handles feature generation."""
import spacy
from nltk.util import skipgrams
from nltk import ngrams
from collections import Counter, OrderedDict
from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from z_api.utils.cleaners import DocumentCleaner


class LinguisticFeatures(object):
    """Linguistic feature generation class."""

    def __init__(self, methods: List[str] = None, cleaner: DocumentCleaner = None,
                 **kwargs):
        """Set initialisations so that loading only happens once."""
        # Initialise variables
        self.kwargs       = kwargs
        self.method     = methods
        self.method_map = OrderedDict()
        self.features   = {}

        # Initialise imports
        self.tagger = spacy.load('en')
        self.dc     = cleaner
        self.sent   = SentimentIntensityAnalyzer()

    @property
    def methods(self) -> dict:
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
    def doc(self) -> str:
        """Document handler."""
        return self.dc.document

    @doc.setter
    def doc(self, document) -> None:
        self.dc.doc = document
        self.dc.generate()

    def generate(self) -> dict:
        """Generate features, where each item is a callable function."""
        if self.method_map == {}:
            self.methods = self.method

        for m_str in self.method_map:
            res = self.method_map[m_str](**self.kwargs)
            if res:
                self.features.update(Counter(res))
        return self.features

    def unigrams(self, **kwargs) -> List[str]:
        """Return unigrams after removal of stopwords."""
        return self.dc.tokens

    def token_ngrams(self, **kwargs) -> List[str]:
        """Generate list of token n-grams, n given in kwargs['ngrams'].

        :param kwargs: Keyword Args (must contain 'ngrams').
        :return: list[str]: Multi-token tokens joined by _.
        e.g.: Outstanding blossom -> Outstanding_blossom
        """
        return ["_".join(tok) for tok in ngrams(self.dc.current, kwargs['ngrams'])]

    def skip_grams(self, **kwargs) -> List[str]:
        """Generate list of skip-grams.

        :param kwargs: Keyword Args (must contain 'ngrams' and 'skip_size').
        :return: list[str]: Multi-token tokens joined by _.
        """
        return ["_".join(item) for item in skipgrams(self.dc.current,
                kwargs['ngrams'], kwargs['skip_size'])]

    def char_ngrams(self, **kwargs) -> List[str]:
        """Generate list of character n-grams.

        :param kwargs: Keyword Args (must contain 'char-ngrams').
        :return: list[str]: Multi-token tokens joined by _.
        """
        return ["_".join(toks) for toks in ngrams(" ".join(self.dc.current), kwargs['char_ngrams'])]

    def sentiment_aggregate(self, **kwargs) -> None:
        """Compute sentiment aggregate and directly update features dictionary."""
        sent = self.sent.polarity_scores(self.dc.document)

        if sent['compound'] >= 0.5:
            update = {'SENTIMENT': 'pos'}
        elif sent['compound'] > -0.5 and sent['compound'] < 0.5:
            update = {'SENTIMENT': 'neu'}
        else:
            update = {'SENTIMENT': 'neg'}

        if 'test' in self.kwargs:
            return update
        else:
            self.features.update(update)

    def sentiment_scores(self, **kwargs) -> None:
        """Compute sentiment scores and directly update features dictionary."""
        sent = self.sent.polarity_scores(self.dc.document)
        del sent['compound']
        self.features.update(sent)
        if 'test' in self.kwargs:
            return sent

    def word_count(self, **kwargs) -> dict:
        """Compute the number of words in the document. Directly update feature dict.

        :param stopped: bool: Use stopword filtered text.
        """
        self.features.update({'TOK_COUNT': len(self.dc.current)})

        if 'test' in self.kwargs:
            return {"TOK_COUNT": len(self.dc.current)}

    def avg_word_length(self, **kwargs):
        """Compute the average word length in the document. Directly update feature dict.

        :param stopped: bool: Use stopword filtered text.
        """
        tok_len = sum(len(w) for w in self.dc.current) / len(self.dc.current)

        self.features.update({'AVG_TOK_LEN': round(tok_len,2)})

        if 'test' in self.kwargs:
            return {"AVG_TOK_LEN": round(tok_len,2)}
