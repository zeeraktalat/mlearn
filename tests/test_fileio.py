import os
import csv
import unittest
from mlearn.base import Field
from mlearn.data import fileio as io
from mlearn.utils.metrics import Metrics
from mlearn.modeling.embedding import RNNClassifier
from mlearn.data.dataset import GeneralDataset


class TestFileIO(unittest.TestCase):
    """Test FileIO.py."""

    @classmethod
    def tearDown(cls):
        """Tear down objects."""
        if os.path.exists('test'):
            os.remove('test')

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

            dev_scores = Metrics(['accuracy'], 'accuracy', 'accuracy')
            dev_scores.scores = {'accuracy': [0.2], }

            model = RNNClassifier(100, 50, 25, 2, 0.2)
            model_hdr = ['Dropout', 'Model', 'Input dim', 'Embeding dim', 'Hidden dim', 'Output dim']
            hyper_info = ['# Epochs', 'Learning rate', 'Batch size']

            self.assertTrue(io.write_results(writer = writer, model = model, model_hdr = model_hdr,
                                             data_name = 'test', main_name = 'test222', hyper_info = hyper_info,
                                             metric_hdr = train_scores.scores.keys(), metrics = train_scores,
                                             dev_metrics = dev_scores),
                            msg = "Successful run of write_results failed.")

            with self.assertRaises(KeyError, msg = "Does not invoke Exceptions."):
                # Key error
                train_scores.scores = {'precision': 0.2}
                io.write_results(writer = writer, model = model, model_hdr = model_hdr,
                                 data_name = 'test', main_name = 'test222', hyper_info = hyper_info,
                                 metric_hdr = train_scores.scores.keys(), metrics = train_scores,
                                 dev_metrics = dev_scores),

    def test_write_predictions(self):
        """Test fileio.write_predictions()"""
        fields = [Field('text', train = True, label = False, ignore = False, ix = 0, cname = 'text'),
                  Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 1)]

        dataset = GeneralDataset(data_dir = os.getcwd() + '/tests/data/',
                                 ftype = 'csv', fields = fields, train = 'train.csv', dev = 'dev.csv',
                                 test = 'test.csv', train_labels = None, tokenizer = lambda x: x.split(),
                                 preprocessor = None, transformations = None, label_processor = None,
                                 sep = ',', name = 'test')
        dataset.load('train')
        dataset.itol = {}

        for obj in dataset.data:
            setattr(obj, 'pred', 0)

        with open('test', 'w', encoding = 'utf-8') as inf:
            writer = csv.writer(inf, delimiter = '\t')
            model = RNNClassifier(100, 50, 25, 2, 0.2)
            model_hdr = ['Dropout', 'Model', 'Input dim', 'Embeding dim', 'Hidden dim', 'Output dim']
            hyper_info = ['# Epochs', 'Learning rate', 'Batch size']

            with self.assertRaises((IndexError, KeyError), msg = "Exception not raised."):
                io.write_predictions(writer, model, model_hdr, 'test1', 'test2', hyper_info,
                                     dataset.data, dataset, 'text', 'label')

            dataset.build_token_vocab(dataset.data)
            dataset.build_label_vocab(dataset.data)
            for obj in dataset.data:
                setattr(obj, 'label', dataset.label_name_lookup(obj.label))

            self.assertTrue(io.write_predictions(writer, model, model_hdr, 'test1', 'test2', hyper_info,
                                                 dataset.data, dataset, 'text', 'label'),
                            msg = "Writing of predictions file failed.")
