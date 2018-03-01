from typing import Union, List, Dict, Tuple, Callable
from collections import Counter, defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import ngrams, word_tokenize
from nltk.util import skipgrams
import spacy

class LinguisticFeatures(object):
    def __init__(self, methods: List[str] = [], cleaner: Callable = None, **kwargs):
        # Initialise variables
        self.args       = kwargs
        self.methods    = methods
        self.method_map = {}
        self.features   = {}

        # Initialise imports
        self.tagger  = spacy.load('en')
        self.cleaner = cleaner
        self.sent    = SentimentIntensityAnalyzer()

