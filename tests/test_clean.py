import unittest
from mlearn.data.clean import Preprocessors, Cleaner


class TestCleaner(unittest.TestCase):

    @classmethod
    def setUp(cls):
        """Set up class with data as well."""
        cls.original = "This is written by @zeerakw about #Onlineabuse on http://www.google.com"
        cls.clean = Cleaner()

    def test_clean_document(self):
        """Test document cleaning."""
        expected = "this is written by AT_USER about HASHTAG on URL"
        result = self.clean.clean_document(self.original, ['lower', 'url', 'hashtag', 'username'])
        self.assertEqual(expected, result, msg = "Document cleaning failed.")

    def test_tokenize(self):
        """Test tokenize function."""
        expected = "This is written by @zeerakw about # Onlineabuse on http://www.google.com".split()
        result = self.clean.tokenize(self.original)
        self.assertListEqual(expected, result, msg = "Document tokenisation failed.")


class TestPreprocessor(unittest.TestCase):

    @classmethod
    def setUp(cls):
        """Set up class with data as well."""
        cls.original = "This is written by @zeerakw about #Onlineabuse on http://www.google.com"
        cls.preprocess = Preprocessors('tests/data/')

    def test_length_tokenize(self):
        """Test the word length."""
        expected = [len(tok) for tok in self.original.split()]
        result = self.preprocess.word_length(self.original.split())
        self.assertListEqual(expected, result, msg = "Length tokenisation failed.")

    def test_PTB_tokenize(self):
        """Test tokenisation using PTB tags."""
        expected = ['DT', 'VBZ', 'VBN', 'IN', 'NN', 'IN', '$', 'NN', 'IN', 'ADD']
        result = self.preprocess.ptb_tokenize(self.original.split(), ['lower', 'url', 'hashtag', 'username'])
        self.assertEqual(expected, result, msg = "PTB tokenisation failed.")

    def test_word_tokenize(self):
        """Test word tokenisation."""
        expected = self.original.split()
        result = self.preprocess.word_token(self.original.split())
        self.assertListEqual(expected, result, msg = "Word tokenisation failed.")

    def test_syllable_count_tokenize(self):
        """Test syllable count tokenisation."""
        expected = [1, 1, 2, 1, 2, 2, 4, 1, 3]
        result = self.preprocess.syllable_count(self.original.split())
        self.assertListEqual(expected, result, msg = "Syllable Count tokenisation failed.")

    def test_liwc_tokenize(self):
        """Test computation of LIWC tokenisation."""
        expected = ['FUNCTION_IPRON_PRONOUN', 'AUXVERB_FUNCTION_VERB_FOCUSPRESENT', 'FOCUSPAST_WORK_VERB',
                    'FUNCTION_PREP', 'USER', 'FUNCTION_ADVERB_PREP', 'AFFECT_NEGEMO_ANGER',
                    'RELATIV_FUNCTION_SPACE_PREP', 'UNK']
        result = self.preprocess.compute_unigram_liwc(self.original.lower().split())
        self.assertListEqual(expected, result, msg = "List-based unigram LIWC tokenization failed.")

        result = self.preprocess.compute_unigram_liwc(self.original.lower())
        self.assertListEqual(expected, result, msg = "Str-based unigram LIWC tokenization failed.")
