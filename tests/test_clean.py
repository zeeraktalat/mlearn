import os
import unittest
from mlearn.base import Field
from mlearn.data_processing.data import GeneralDataset
from mlearn.data_processing.clean import Preprocessors, Cleaner


class TestCleaner(unittest.TestCase):

    @classmethod
    def setUp(cls):
        """Set up class with data as well."""
        fields = [Field('text', train = True, label = False, ignore = False, ix = 0, cname = 'text'),
                  Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 1)]

        cls.dataset = GeneralDataset(data_dir = os.getcwd() + '/tests/',
                                         ftype = 'csv', fields = fields, train = 'train.csv', dev = None,
                                         test = 'test.csv', train_labels = None, tokenizer = lambda x: x.split(),
                                         preprocessor = None, transformations = None,
                                         label_processor = None, sep = ',', name = 'test')
        cls.dataset.load('train')
        cls.train = cls.dataset.data
        cls.clean = Cleaner()

    @classmethod
    def tearDown(cls):
        """Take down class."""
        cls.dataset = 0

    def test_clean_document(self):
        """Test document cleaning."""
        original = "This is written by @zeerakw about #Onlineabuse on http://www.google.com"
        expected = "This is written by AT_USER about HASHTAG on URL"
        result = self.clean.clean_document(original, ['lower', 'url', 'hashtag', 'username'])
        self.assertEqual(expected, result, msg = "Document cleaning failed.")

    def test_tokenize(self):
        """Test tokenize function."""
        original = "This is written by @zeerakw about #Onlineabuse on http://www.google.com".split()
        expected = "This is written by @zeerakw about # Onlineabuse on http://www.google.com".split()
        result = self.clean.tokenize(original)
        self.assertEqual(expected, result, msg = "Document tokenisation failed.")


class TestPreprocessor(unittest.TestCase):

    @classmethod
    def setUp(cls):
        """Set up class with data as well."""
        fields = [Field('text', train = True, label = False, ignore = False, ix = 0, cname = 'text'),
                  Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 1)]

        cls.csv_dataset = GeneralDataset(data_dir = os.getcwd() + '/tests/',
                                         ftype = 'csv', fields = fields, train = 'train.csv', dev = None,
                                         test = 'test.csv', train_labels = None, tokenizer = lambda x: x.split(),
                                         preprocessor = None, transformations = None,
                                         label_processor = None, sep = ',', name = 'test')
        cls.csv_dataset.load('train')
        cls.train = cls.csv_dataset.data
        cls.preprocess = Preprocessors()

    @classmethod
    def tearDown(cls):
        """Take down class."""
        cls.csv_dataset = 0

    @unittest.skip("Test not implemented.")
    def test_word_length(self):
        """Test the word length."""
        pass

    def test_PTB_tokenize(self):
        """Test tokenisation using PTB tags."""
        original = "This is written by @zeerakw about #Onlineabuse on http://www.google.com"
        expected = ['DT', 'VBZ', 'VBN', 'IN', 'NN', 'IN', '$', 'NN', 'IN', 'ADD']
        result = self.clean.clean_document(original, ['lower', 'url', 'hashtag', 'username'])
        self.assertEqual(expected, result, msg = "Document cleaning failed.")
