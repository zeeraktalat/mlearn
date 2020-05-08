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

        cls.csv_dataset = GeneralDataset(data_dir = os.getcwd() + '/tests/',
                                         ftype = 'csv', fields = fields, train = 'train.csv', dev = None,
                                         test = 'test.csv', train_labels = None, tokenizer = lambda x: x.split(),
                                         preprocessor = None, transformations = None,
                                         label_processor = None, sep = ',', name = 'test')
        cls.csv_dataset.load('train')
        cls.train = cls.csv_dataset.data
        cls.clean = Cleaner()

    @classmethod
    def tearDown(cls):
        """Take down class."""
        cls.csv_dataset = 0

    @unittest.skip("Test not implemented.")
    def test_tokenize(self):
        """Test tokenize function."""
        pass


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
