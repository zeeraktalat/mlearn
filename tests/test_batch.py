import os
import torch
import torchtestcase
from mlearn.base import Field, Datapoint
from mlearn.data.dataset import GeneralDataset
from mlearn.data.batching import Batch, BatchExtractor


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
        """Test Batch.create_batches()."""
        batches = Batch(32, self.train)
        batches.create_batches()
        self.assertIsNotNone(batches)
        counts = sum(len(b) for b in batches)
        self.assertEqual(counts, len(self.train))

    def test_batch_count(self):
        """Test Batch.__len__()."""
        b = Batch(32, self.train)
        b.create_batches()
        expected = 15
        self.assertEqual(len(b), expected, msg = "The number of batches is wrong.")

    def test_batch_contents(self):
        """Test Batch contains Datapoint."""
        b = Batch(32, self.train)
        b.create_batches()
        expected = [True for batch in b]
        output = [all(isinstance(batch_item, Datapoint) for batch_item in batch) for batch in b]
        self.assertEqual(output, expected, msg = "Not all items are datapoints.")

    def test_shuffle_batches(self):
        """Test Batch.shuffle_batches()."""
        batches = Batch(32, self.train)
        batches.create_batches()
        expected = [batch for batch in batches]
        result = [batch for batch in batches.shuffle_batches()]
        self.assertCountEqual(expected, result, msg = "Shuffling batches failed: Contents in the batches have changed.")
        self.assertFalse(expected == result, msg = "Shuffling batches failed: Batches have not been shuffled.")

    def test_shuffle_data(self):
        """Test Batch.shuffle()."""
        batches = Batch(32, self.train)
        batches.create_batches()
        expected = [item for batch in batches for item in batch]
        result = [item for batch in batches.shuffle() for item in batch]
        self.assertCountEqual(expected, result, msg = "Shuffling data failed: Contents of the batches have changed.")
        self.assertFalse(expected == result, msg = "Shuffling data failed: Content has not been changed.")

    def test_getitem__(self):
        """Test Batch.__getitem__ function."""
        batches = Batch(32, self.train)
        batches.create_batches()
        expected = next(iter(batches))
        result = batches[0]
        self.assertListEqual(expected, result, msg = "Batch.__getitem__ does not work.")

    # def test_getattr__(self):
    #     """Test Batch.__getattr__ function."""
    #     batches = Batch(32, self.train)
    #     batches.create_batches()
    #     expected = [item.text for batch in batches for item in batch]
    #     result = list(getattr(batches, 'text'))
    #     self.assertListEqual(expected, result, msg = "Batch.__getattr__ does not provide the correct return value.")


class TestBatchGenerator(torchtestcase.TorchTestCase):
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
        """Test BatchExtractor.__len__."""
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

    def test_getitem__(self):
        """Test BatchExtractor.__getitem__ function."""
        batches = BatchExtractor('label', self.batches, self.dataset, onehot = False)
        expected = next(iter(batches))
        result = batches[0]

        expected_ys = [exp[1] for exp in expected]
        result_ys = [tup[1] for tup in result]
        self.assertEqual(expected_ys[0], result_ys[0], msg = "Batch.__getitem__ labels do not match.")

        expected_xs = [exp[0] for exp in expected]
        result_xs = [tup[0] for tup in result]
        self.assertEqual(expected_xs[0], result_xs[0], msg = "Batch.__getitem__ tensors do not match.")

    def test_shuffle_batches(self):
        """Test BatchExtractor.shuffle_batches()."""
        batches = BatchExtractor('label', self.batches, self.dataset, onehot = False)
        expected = [batch for batch in batches]
        result = [batch for batch in batches.shuffle_batches()]

        expected_ys = [exp[1].tolist() for exp in expected]
        result_ys = [tup[1].tolist() for tup in result]
        results = [lab == res for lab, res in zip(expected_ys, result_ys)]
        self.assertFalse(all(results), msg = "BatchExtractor.__getitem__ batch are not shuffled.")

    def test_shuffle_data(self):
        """Test shuffling data."""
        batches = BatchExtractor('label', self.batches, self.dataset, onehot = False)
        expected = [batch for batch in batches]
        result = [batch for batch in batches.shuffle_batches()]

        expected_ys = []
        result_ys = []

        for exp, res in zip(expected, result):
            expected_ys.extend(exp[1].tolist())
            result_ys.extend(res[1].tolist())

        results = [lab == res for lab, res in zip(expected_ys, result_ys)]
        self.assertFalse(all(results), msg = "BatchExtractor.__getitem__ labels are not shuffled.")
