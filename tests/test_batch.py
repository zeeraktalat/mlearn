import os
import torch
import unittest
import torchtestcase
from mlearn.base import Field, Datapoint
from mlearn.data_processing.data import GeneralDataset
from mlearn.data_processing.batching import Batch, BatchExtractor


class TestBatch(torchtestcase.TorchTestCase):
    """Test the Batch class."""

    @classmethod
    def setUp(cls):
        """Set up necessary class variables."""
        text_field = Field('text', train = True, label = False, ignore = False, ix = 5, cname = 'text')
        label_field = Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 4)
        ignore_field = Field('ignore', train = False, label = False, cname = 'ignore', ignore = True)

        fields = [ignore_field, ignore_field, ignore_field, ignore_field, text_field, label_field]

        cls.dataset = GeneralDataset(data_dir = os.getcwd() + '/tests/',
                                     ftype = 'csv', fields = fields, train = 'garcia_stormfront_test.tsv', dev = None,
                                     test = None, train_labels = None, tokenizer = lambda x: x.split(),
                                     preprocessor = None, transformations = None,
                                     label_processor = None, sep = '\t', name = 'test')
        cls.dataset.load('train')
        cls.train = cls.dataset.data

    @classmethod
    def tearDown(cls):
        """Tear down class between each test."""
        cls.dataset = 0

    def test_batch_sizes(self):
        """Test that the entire dataset has been batched."""
        batches = Batch(32, self.train)
        batches.create_batches()
        self.assertIsNotNone(batches)
        counts = sum(len(b) for b in batches)
        self.assertEqual(counts, len(self.train))

    def test_batch_count(self):
        """Test that each batch contains datapoints."""
        b = Batch(32, self.train)
        b.create_batches()
        expected = 15
        self.assertEqual(len(b), expected, msg = "The number of batches is wrong.")

    def test_batch_contents(self):
        """Test that each batch contains datapoints."""
        b = Batch(32, self.train)
        b.create_batches()
        expected = [True for batch in b]
        output = [all(isinstance(batch_item, Datapoint) for batch_item in batch) for batch in b]
        self.assertEqual(output, expected, msg = "Not all items are datapoints.")

    @unittest.skip("Test is not implemented.")
    def test_shuffle_batches(self):
        """Test shuffling batches."""
        pass

    @unittest.skip("Test is not implemented.")
    def test_shuffle_data(self):
        """Test shuffling data."""
        pass

    @unittest.skip("Test not implemented.")
    def test_getitem__(self):
        """Test __getitem__ function."""
        pass

    @unittest.skip("Test not implemented.")
    def test_getattr__(self):
        """Test __getattr__ function."""
        pass


class TestBatchGenerator(unittest.TestCase):
    """Test the batchgenerator class."""

    @classmethod
    def setUp(cls):
        """Set up necessary class variables."""
        text_field = Field('text', train = True, label = False, ignore = False, ix = 5, cname = 'text')
        label_field = Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 4)
        ignore_field = Field('ignore', train = False, label = False, cname = 'ignore', ignore = True)

        fields = [ignore_field, ignore_field, ignore_field, ignore_field, text_field, label_field]

        cls.dataset = GeneralDataset(data_dir = os.getcwd() + '/tests/',
                                     ftype = 'csv', fields = fields, train = 'garcia_stormfront_test.tsv', dev = None,
                                     test = None, train_labels = None, tokenizer = lambda x: x.split(),
                                     preprocessor = None, transformations = None,
                                     label_processor = None, sep = '\t', name = 'test')
        # Load
        cls.dataset.load('train')
        train = cls.dataset.data

        #
        cls.dataset.build_token_vocab(train)
        cls.dataset.build_label_vocab(train)
        cls.dataset.process_labels(train)

        cls.batch_size = 64
        cls.batches = Batch(cls.batch_size, train)
        cls.batches.create_batches()

    @classmethod
    def tearDown(cls):
        """Tear down class between each test."""
        cls.dataset = 0
        cls.batches = 0

    def test_batch_generation(self):
        """Test the batchgenerator can access the variables."""
        batches = BatchExtractor('label', self.batches, self.dataset)

        for batch in batches:
            self.assertEqual(batch[0].size(0), batch[1].size(0))

    def test_tensorisation(self):
        """Test that the batches are all tensorized."""
        batches = BatchExtractor('label', self.batches, self.dataset)

        for batch in batches:
            self.assertIsInstance(batch[0], torch.Tensor, msg = "The type of the data element is incorrect.")
            self.assertIsInstance(batch[1], torch.Tensor, msg = "The type of the label element is incorrect.")

    def test_batch_sizes(self):
        """Test the __len__ function of a BatchExtractor."""
        batches = BatchExtractor('label', self.batches, self.dataset)
        self.assertEqual(len(batches), 8, msg = "The len operation on BatchExtractor is incorrect.")

    def test_onehot_encoded_batches(self):
        """Test creation of onehot encoded batches."""
        batches = BatchExtractor('label', self.batches, self.dataset, onehot = True)

        expected = [self.dataset.vocab_size() for batch in batches]
        output = [batch[0].size(2) for batch in batches]
        self.assertEqual(output, expected, msg = "Not all encoded items have the right length.")

    def test_index_encoded_batches(self):
        """Test creation of index encoded batches."""
        batches = BatchExtractor('label', self.batches, self.dataset, onehot = False)

        expected = [self.dataset.length for _ in batches]
        output = [batch[0].size(1) for batch in batches]
        self.assertEqual(output, expected, msg = "Not all encoded items have the right length.")

    @unittest.skip("Test not implemented.")
    def test_getitem__(self):
        """Test __getitem__ function."""
        pass

    @unittest.skip("Test not implemented.")
    def test_getattr__(self):
        """Test __getattr__ function."""
        pass

    @unittest.skip("Test is not implemented.")
    def test_shuffle_batches(self):
        """Test shuffling batches."""
        pass

    @unittest.skip("Test is not implemented.")
    def test_shuffle_data(self):
        """Test shuffling data."""
        pass
