import os
import unittest
from mlearn.base import Field, Datapoint
from mlearn.data_processing.data import GeneralDataset


class TestDataPoint(unittest.TestCase):

    @classmethod
    def setUp(cls):
        """Set up class with data as well."""
        fields = [Field('text', train = True, label = False, ignore = False, ix = 0, cname = 'text'),
                  Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 1)]

        cls.dataset = GeneralDataset(data_dir = os.getcwd() + 'tests/',
                                     ftype = 'csv', fields = fields, train = 'train.csv', dev = None,
                                     test = 'test.csv', train_labels = None, tokenizer = lambda x: x.split(),
                                     preprocessor = None, transformations = None,
                                     label_processor = None, sep = ',', name = 'test')
        cls.dataset.load('train')
        cls.train = cls.dataset.data

    def test_datapoint_creation(self):
        """Test that datapoints are created consistently."""
        expected = [{'text': 'me gusta comer en la cafeteria'.lower().split(),
                     'label': 'SPANISH',
                     'original': 'me gusta comer en la cafeteria'},
                    {'text': 'Give it to me'.lower().split(),
                     'label': 'ENGLISH',
                     'original': 'Give it to me'},
                    {'text': 'No creo que sea una buena idea'.lower().split(),
                     'label': 'SPANISH',
                     'original': 'No creo que sea una buena idea'},
                    {'text': 'No it is not a good idea to get lost at sea'.lower().split(),
                     'label': 'ENGLISH',
                     'original': 'No it is not a good idea to get lost at sea'}
                    ]
        for exp, out in zip(expected, self.train):
            self.assertDictEqual(exp, out.__dict__, msg = "A dictionary is not created right.")
            self.assertIsInstance(out, Datapoint)

    def test_datapoint_counts(self):
        """Test the correct number of datapoints are created."""
        self.assertEqual(4, len(self.train))
