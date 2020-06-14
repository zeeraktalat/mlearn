import os
import csv
import unittest
from mlearn.base import Field
from mlearn.data import fileio as io
from mlearn.utils.metrics import Metrics
from mlearn.data.dataset import GeneralDataset


class TestFileIO(unittest.TestCase):
    """Test FileIO.py."""

    def test_read_json(self):
        """Test fileio.read_json()"""
        filepath = 'tests/data/traindeep.json'
        doc_key, label_key = 'text', 'label'
        expected = [{'text': 'me gusta comer en la cafeteria', 'label': 'spanish', 'username': 'zeerakw',
                     'name': 'zeerak', 'place': 'some place'},
                    {'text': 'give it to me', 'label': 'english', 'username': 'madeup',
                     'name': 'made', 'place': 'far far away'},
                    {'text': 'no creo que sea una buena idea', 'label': 'spanish', 'username': 'upmade',
                     'name': 'up', 'place': 'long lost'},
                    {'text': 'no it is not a good idea to get lost at sea', 'label': 'english', 'username': 'notreal',
                     'name': 'unreal', 'place': 'rock and hard place'}
                    ]
        result = io.read_json(filepath, 'utf-8', doc_key, label_key,
                              secondary_keys = {'username': 'user|username',
                                                'name': 'user|name',
                                                'place': 'user|location|place'})
        for res, exp in zip(result, expected):
            self.assertDictEqual(res, exp, msg = "A dictionary is not loaded correctly.")

    def test_write_results(self):
        """Test fileio.write_results()"""
        fields = [Field('text', train = True, label = False, ignore = False, ix = 0, cname = 'text'),
                  Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 1)]

        dataset = GeneralDataset(data_dir = os.getcwd() + '/tests/data/',
                                 ftype = 'csv', fields = fields, train = 'train.csv', dev = 'dev.csv',
                                 test = 'test.csv', train_labels = None, tokenizer = lambda x: x.split(),
                                 preprocessor = None, transformations = None, label_processor = None,
                                 sep = ',', name = 'test')
        dataset.load('train')

        with open('test', 'w', encoding = 'utf-8') as inf:
            writer = csv.writer(inf, delimiter = '\t')
            train_scores = Metrics(['accuracy'], 'accuracy', 'accuracy')
            train_scores.scores = {'accuracy': [0.5]}
            train_loss = 0.2

            dev_scores = Metrics(['accuracy'], 'accuracy', 'accuracy')
            dev_scores.scores = {'accuracy': [0.2]}
            dev_loss = [0.2]

            model_info = ['NotModel', 200, 300]

            self.assertTrue(io.write_results(writer = writer, train_scores = train_scores, train_loss = train_loss,
                                             dev_scores = dev_scores, dev_loss = dev_loss, epochs = 1,
                                             model_info = model_info, metrics = train_scores, exp_len = 10,
                                             data_name  = 'test', main_name = 'test222'),
                            msg = "Successful run of write_results failed.")

            with self.assertRaises((IndexError, KeyError, AssertionError), msg = "Does not invoke errors."):
                # Index Error
                io.write_results(writer = writer, train_scores = train_scores, train_loss = train_loss,
                                 dev_scores = dev_scores, dev_loss = dev_loss, epochs = 20,
                                 model_info = model_info, metrics = train_scores, exp_len = 10,
                                 data_name  = 'test', main_name = 'test222')

                # Assertion Error
                io.write_results(writer = writer, train_scores = train_scores, train_loss = train_loss,
                                 dev_scores = dev_scores, dev_loss = dev_loss, epochs = 20,
                                 model_info = model_info, metrics = train_scores, exp_len = 7,
                                 data_name  = 'test', main_name = 'test222')

                # Key error
                train_scores.scores = {'precision': 0.2}
                io.write_results(writer = writer, train_scores = train_scores, train_loss = train_loss,
                                 dev_scores = dev_scores, dev_loss = dev_loss, epochs = 1,
                                 model_info = model_info, metrics = train_scores, exp_len = 10,
                                 data_name  = 'test', main_name = 'test222')
            os.remove('tests')

    def test_write_predictions(self):
        """Test fileio.write_predictions()"""
        pass
