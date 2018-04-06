import unittest
from mlapi.features.Features import LinguisticFeatures
from mlapi.utils.cleaners import DocumentCleaner

class FeatureTest(unittest.TestCase):
    """Test features."""

    @classmethod
    def setUpClass(cls):
        """Setup class."""
        methods = [
                   'unigrams',
                   'token_ngrams',
                   'skip_grams',
                   'char_ngrams',
                   'sentiment',
                   'word_count',
                   'avg_word_length'
                   ]
        dc = DocumentCleaner(['nltk_tokenize'])
        kwargs = {
                  'ngrams': 2,
                  'char_ngrams': 3,
                  'skip_size': 2,
                  'stopped': False,
                  }
        cls.fc = LinguisticFeatures(methods, dc, **kwargs)

    @classmethod
    def tearDownClass(cls):
        """Tear down information from class."""
        pass

    def test_unigrams(self):
        pass

    def test_token_ngrams(self):
        pass

    def test_skip_grams(self):
        pass

    def test_char_ngrams(self):
        pass

    def test_sentiment(self):
        pass

    def test_word_count(self):
        pass

    def test_avg_word_length(self):
        pass
