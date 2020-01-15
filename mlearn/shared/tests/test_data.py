import torch
import unittest
import torchtestcase
from ..data import GeneralDataset
from ..base import Field, Datapoint
from ..batching import Batch, BatchExtractor


class TestDataSet(torchtestcase.TorchTestCase):

    @classmethod
    def setUpClass(cls):
        """Set up class with data as well."""
        fields = [Field('text', train = True, label = False, ignore = False, ix = 0, cname = 'text'),
                  Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 1)]

        cls.csv_dataset = GeneralDataset(data_dir = '~/PhD/projects/active/Generalisable_abuse/gen/shared/tests/',
                                         ftype = 'csv', fields = fields, train = 'train.csv', dev = None,
                                         test = 'test.csv', train_labels = None, tokenizer = lambda x: x.split(),
                                         preprocessor = None, transformations = None,
                                         label_processor = None, sep = ',')
        cls.csv_dataset.load('train')
        cls.train = cls.csv_dataset.data
        cls.json_dataset = GeneralDataset(data_dir = '~/PhD/projects/active/Generalisable_abuse/gen/shared/tests/',
                                          ftype = 'json', fields = fields, train = 'train.json', dev = None,
                                          test = 'test.json', train_labels = None, tokenizer = lambda x: x.split(),
                                          preprocessor = None, transformations = None,
                                          label_processor = None, sep = ',')

    @classmethod
    def tearDownClass(cls):
        """Take down class."""
        cls.csv_dataset = 0
        cls.json_dataset = 0

    def test_load(self):
        """Test dataset loading."""
        expected = [("me gusta comer en la cafeteria".lower().split() + ['<pad>'] * 6, "SPANISH"),
                    ("Give it to me".lower().split() + ['<pad>'] * 8, "ENGLISH"),
                    ("No creo que sea una buena idea".lower().split() + ['<pad>'] * 5, "SPANISH"),
                    ("No it is not a good idea to get lost at sea".lower().split(), "ENGLISH")]

        csv_train = self.train
        output = [(doc.text, doc.label) for doc in csv_train]
        self.assertListEqual(output, expected, msg = 'Data Loading failed.')

        self.json_dataset.load('train', skip_header = False)
        json_train = self.json_dataset.data
        output = [(doc.text, doc.label) for doc in json_train]
        self.assertListEqual(output, expected, msg = 'Data Loading failed.')
        self.assertIsInstance(json_train[0], Datapoint, msg = 'Data Loading failed gave wrong type.')

    def test_build_token_vocab(self):
        """Test vocab building method."""
        expected = set(['<pad>', '<unk>'] + list(sorted("""me gusta comer en la cafeteria Give it to me
                   No creo que sea una buena idea No it is not a good idea to get lost at sea""".lower().split())))
        self.csv_dataset.build_token_vocab(self.train)
        output = set(sorted(self.csv_dataset.stoi.keys()))
        self.assertSetEqual(output, expected, msg = 'Vocab building failed.')

    def test_extend_vocab(self):
        """Test extending vocab."""
        train = """<pad> <unk> me gusta comer en la cafeteria Give it to me
                No creo que sea una buena idea No it is not a good idea to get lost at sea""".lower().split()
        test = "Yo creo que si it is lost on me".lower().split()
        expected = set(train + test)
        self.csv_dataset.load('test')
        test = self.csv_dataset.test
        self.csv_dataset.build_token_vocab(self.train)
        self.csv_dataset.extend_vocab(test)
        output = list(self.csv_dataset.stoi.keys())
        self.assertListEqual(sorted(output), sorted(expected), msg = 'Vocab Extension Failed.')

    def test_load_test_from_different_file(self):
        """Test loading a secondary dataset (test/dev set) from a different file."""
        self.csv_dataset.load('test')
        test = self.csv_dataset.test
        expected  = ["Yo creo que si".lower().split() + 8 * ['<pad>'],
                     "it is lost on me".lower().split() + ['<pad>'] * 7]
        self.assertListEqual(test[0].text, expected[0])
        self.assertListEqual(test[1].text, expected[1])

    def test_vocab_token_lookup(self):
        '''Test looking up in vocab.'''
        self.csv_dataset.build_label_vocab(self.train)
        expected = 0
        output = self.csv_dataset.vocab_token_lookup('me')
        self.assertEqual(output, expected, msg = 'Vocab token lookup failed.')

    def test_vocab_ix_lookup(self):
        '''Test looking up in vocab.'''
        self.csv_dataset.build_label_vocab(self.train)
        expected = 'me'
        output = self.csv_dataset.vocab_ix_lookup(0)
        self.assertEqual(output, expected, msg = 'Vocab ix lookup failed.')

    def test_vocab_size(self):
        """Test vocab size is expected size."""
        self.csv_dataset.build_token_vocab(self.train)
        output = self.csv_dataset.vocab_size()
        expected = 25
        self.assertEqual(output, expected, msg = 'Building vocab failed.')

    def test_vocab_limiter(self):
        """Test vocab limiter."""
        self.csv_dataset.build_token_vocab(self.train)

        def limiter(vocab, n = 2):
            return {ix: tok for ix, (tok, c) in enumerate(vocab.most_common()) if c >= n}

        self.csv_dataset.limit_vocab(limiter, n = 2)
        output = [tok for tok in self.csv_dataset.stoi]
        expected = ['me', 'it', 'to', 'no', 'sea', 'idea', '<pad>', '<unk>']
        self.assertListEqual(output, expected, msg = "Limiting vocab failed.")

    def test_build_label_vocab(self):
        """Test building label vocab."""
        self.csv_dataset.build_label_vocab(self.train)
        output = list(sorted(self.csv_dataset.ltoi.keys()))
        expected = ['ENGLISH', 'SPANISH']
        self.assertListEqual(output, expected, msg = 'Building label vocab failed.')

    def test_label_name_lookup(self):
        """Test looking up in label."""
        self.csv_dataset.build_label_vocab(self.train)
        output = self.csv_dataset.label_name_lookup('SPANISH')
        expected = 1
        self.assertEqual(output, expected, msg = 'label name lookup failed.')

    def test_label_ix_lookup(self):
        '''Test looking up in label.'''
        self.csv_dataset.build_label_vocab(self.train)
        output = self.csv_dataset.label_ix_lookup(1)
        expected = 'SPANISH'
        self.assertEqual(output, expected, msg = 'label ix lookup failed.')

    def test_label_count(self):
        """Test label size is expected."""
        self.csv_dataset.build_label_vocab(self.train)
        expected = self.csv_dataset.label_count()
        output = 2
        self.assertEqual(output, expected, msg = 'Test that label count matches labels failed.')

    def test_process_label(self):
        """Test label processing."""
        self.csv_dataset.build_label_vocab(self.train)
        expected = [1]
        output = self.csv_dataset._process_label('SPANISH')
        self.assertEqual(output, expected, msg = 'Labelprocessor failed without custom processor')

        def processor(label):
            labels = {'SPANISH': 1, 'ENGLISH': 0}
            return labels[label]

        expected = [1]
        output = self.csv_dataset._process_label('SPANISH', processor = processor)
        self.assertEqual(output, expected, msg = 'Labelprocessor failed with custom processor')

    def test_no_preprocessing(self):
        """Test document processing."""
        setattr(self.csv_dataset, 'lower', False)
        setattr(self.csv_dataset, 'preprocessor', None)
        setattr(self.csv_dataset, 'repr_transform', None)

        inputs = "Give it to me, baby. Uhuh! Uhuh!"
        expected = inputs.split()
        output = self.csv_dataset.process_doc(inputs)
        self.assertListEqual(output, expected, msg = 'Process Document failed.')
        self.assertIsInstance(output, list, msg = 'Process document returned wrong type.')

    def test_lower_doc(self):
        """Test lowercasing processing."""
        setattr(self.csv_dataset, 'lower', True)
        setattr(self.csv_dataset, 'preprocessor', None)
        setattr(self.csv_dataset, 'repr_transform', None)

        inputs = "Give it to me, baby. Uhuh! Uhuh!"
        expected = inputs.lower().split()
        output = self.csv_dataset.process_doc(inputs)
        self.assertListEqual(output, expected, msg = 'Process Document failed with lowercasing.')
        self.assertIsInstance(output, list, msg = 'Process document with lowercasing produces wrong type.')

    def test_list_process_doc(self):
        """Test lowercasing processing."""
        setattr(self.csv_dataset, 'lower', True)
        setattr(self.csv_dataset, 'preprocessor', None)
        setattr(self.csv_dataset, 'repr_transform', None)

        inputs = "Give it to me, baby. Uhuh! Uhuh!"
        expected = inputs.lower().split()
        output = self.csv_dataset.process_doc(inputs.split())
        self.assertListEqual(output, expected, msg = 'Process Document failed with input type list.')
        self.assertIsInstance(output, list, msg = 'Process document with input type list produces wrong type.')

    def test_custom_preprocessor(self):
        """Test using a custom processor."""

        inputs = "Give it to me, baby. Uhuh! Uhuh!"
        expected = ["TEST" if '!' in tok else tok for tok in inputs.lower().split()]

        def preprocessor(doc):
            return ["TEST" if '!' in tok else tok for tok in doc]

        setattr(self.csv_dataset, 'lower', True)
        setattr(self.csv_dataset, 'preprocessor', preprocessor)
        output = self.csv_dataset.process_doc(inputs)
        self.assertListEqual(output, expected, msg = 'Process Document failed with preprocessor')
        self.assertIsInstance(output, list, msg = 'Process Document with preprocessor returned wrong type.')

    def test_repr_transformation(self):
        """Test using transformation to different representation."""

        inputs = "Give it to me, baby. Uhuh! Uhuh!"
        expected = "VERB DET PREP PRON NOUN AGREEMENT AGREEMENT".split()

        def transform(doc):
            transform = {'give': 'VERB', 'it': 'DET', 'to': "PREP", 'me': "PRON", 'baby': 'NOUN', 'uhuh!': 'AGREEMENT',
                         'uhuh': 'AGREEMENT'}
            return [transform[w.lower().replace(',', '').replace('.', '')] for w in doc]

        setattr(self.csv_dataset, 'repr_transform', transform)
        output = self.csv_dataset.process_doc(inputs)
        self.assertListEqual(output, expected, msg = 'Process Document failed with represntation transformation.')
        self.assertIsInstance(output, list, msg = 'Process Document with representation transformation returned wrong')

    def test_pad(self):
        """Test padding of document."""
        exp = ["me gusta comer en la cafeteria".split() + 6 * ['<pad>']]
        exp.append(['give', 'it', 'to', 'me'] + 8 * ['<pad>'])
        exp.append(['no', 'creo', 'que', 'sea', 'una', 'buena', 'idea'] + 5 * ['<pad>'])
        exp.append(['no', 'it', 'is', 'not', 'a', 'good', 'idea', 'to', 'get', 'lost', 'at', 'sea'] + 0 * ['<pad>'])
        output = [dp.text for dp in self.csv_dataset.pad(self.train, length = 12)]
        self.assertListEqual(output, exp, msg = 'Padding doc failed.')

    def test_trim(self):
        """Test that trimming of the document works."""
        expected = 0 * ['<pad>'] + ['no', 'it', 'is', 'not', 'a', 'good', 'idea', 'to', 'get', 'lost'][:5]
        output = list(self.csv_dataset.pad(self.train, length = 5))[-1]
        self.assertListEqual(output.text, expected, msg = 'Zero padding failed.')

    def test_onehot_encoding(self):
        """Test the onehot encoding."""
        self.csv_dataset.build_token_vocab(self.train)
        self.csv_dataset.load('test')
        test = self.csv_dataset.test
        expected = torch.zeros(2, 12, 25)
        expected[0][0][23] = 1
        expected[0][1][12] = 1
        expected[0][2][13] = 1
        expected[0][3][23] = 1
        expected[0][4][24] = 1
        expected[0][5][24] = 1
        expected[0][6][24] = 1
        expected[0][7][24] = 1
        expected[0][8][24] = 1
        expected[0][9][24] = 1
        expected[0][10][24] = 1
        expected[0][11][24] = 1
        expected[1][0][1] = 1
        expected[1][1][16] = 1
        expected[1][2][21] = 1
        expected[1][3][23] = 1
        expected[1][4][0] = 1
        expected[1][5][24] = 1
        expected[1][6][24] = 1
        expected[1][7][24] = 1
        expected[1][8][24] = 1
        expected[1][9][24] = 1
        expected[1][10][24] = 1
        expected[1][11][24] = 1

        output = torch.cat([datapoint.encoded for datapoint in self.csv_dataset.encode(test, True)], dim = 0)
        self.assertEqual(output, expected, msg = 'Onehot encoding failed.')

    @unittest.skip("Not Implemented.")
    def test_encoding(self):
        """Test the encoding. Not Implemented."""
        self.csv_dataset.build_token_vocab(self.train)
        self.csv_dataset.load('test')
        test = self.csv_dataset.test
        expected = [[6, 13, 14, 6], [1, 17, 22, 6, 0]]
        output = [datapoint.encoded for datapoint in self.csv_dataset.encode(test, False)]
        self.assertEqual(output, expected, msg = 'Encoding failed.')

    def test_split(self):
        """Test splitting functionality."""
        expected = [3, 1]  # Lengths of the respective splits
        train, test = self.csv_dataset.split(self.train, 0.8)
        output = [len(train), len(test)]
        self.assertListEqual(expected, output, msg = 'Splitting with just int failed.')

        expected = [3, 1]
        train, test = self.csv_dataset.split(self.train, [0.8, 0.2])
        output = [len(train), len(test)]
        self.assertListEqual(expected, output, msg = 'Two split values in list failed.')

        expected = [2, 1, 1]
        train, dev, test = self.csv_dataset.split(self.train, [0.6, 0.2, 0.1])
        output = [len(train), len(dev), len(test)]
        self.assertListEqual(expected, output, msg = 'Three split values in list failed.')


class TestDataPoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up class with data as well."""
        fields = [Field('text', train = True, label = False, ignore = False, ix = 0, cname = 'text'),
                  Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 1)]

        cls.dataset = GeneralDataset(data_dir = '~/PhD/projects/active/Generalisable_abuse/gen/shared/tests/',
                                         ftype = 'csv', fields = fields, train = 'train.csv', dev = None,
                                         test = 'test.csv', train_labels = None, tokenizer = lambda x: x.split(),
                                         preprocessor = None, transformations = None,
                                         label_processor = None, sep = ',')
        cls.dataset.load('train')
        cls.train = cls.dataset.data

    def test_datapoint_creation(self):
        """Test that datapoints are created consistently."""
        expected = [{'text': 'me gusta comer en la cafeteria'.lower().split() + ['<pad>'] * 6,
                     'label': 'SPANISH',
                     'original': 'me gusta comer en la cafeteria'.lower().split()},
                    {'text': 'Give it to me'.lower().split() + ['<pad>'] * 8,
                     'label': 'ENGLISH',
                     'original': 'Give it to me'.lower().split()},
                    {'text': 'No creo que sea una buena idea'.lower().split() + ['<pad>'] * 5,
                     'label': 'SPANISH',
                     'original': 'No creo que sea una buena idea'.lower().split()},
                    {'text': 'No it is not a good idea to get lost at sea'.lower().split(),
                     'label': 'ENGLISH',
                     'original': 'No it is not a good idea to get lost at sea'.lower().split()}
                    ]
        for exp, out in zip(expected, self.train):
            self.assertDictEqual(exp, out.__dict__, msg = "A dictionary is not created right.")
            self.assertIsInstance(out, Datapoint)

    def test_datapoint_counts(self):
        """Test the correct number of datapoints are created."""
        self.assertEqual(4, len(self.train))


class TestBatch(unittest.TestCase):
    """Test the Batch class."""

    @classmethod
    def setUpClass(cls):
        """Set up necessary class variables."""
        text_field = Field('text', train = True, label = False, ignore = False, ix = 6, cname = 'text')
        label_field = Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 5)
        ignore_field = Field('ignore', train = False, label = False, cname = 'ignore', ignore = True)

        fields = [ignore_field, ignore_field, ignore_field, ignore_field, ignore_field, label_field, text_field]

        cls.dataset = GeneralDataset(data_dir = '~/PhD/projects/active/Generalisable_abuse/data/',
                                     ftype = 'csv', fields = fields, train = 'davidson_test.csv', dev = None,
                                     test = None, train_labels = None, tokenizer = lambda x: x.split(),
                                     preprocessor = None, transformations = None,
                                     label_processor = None, sep = ',')
        cls.dataset.load('train')
        cls.train = cls.dataset.data

    @classmethod
    def tearDownClass(cls):
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
        expected = 30
        self.assertEqual(len(b), expected, msg = "The number of batches is wrong.")

    def test_batch_contents(self):
        """Test that each batch contains datapoints."""
        b = Batch(32, self.train)
        b.create_batches()
        expected = [True for batch in b]
        output = [all(isinstance(batch_item, Datapoint) for batch_item in batch) for batch in b]
        self.assertEqual(output, expected, msg = "Not all items are datapoints.")

    def test_onehot_encoded_batches(self):
        """Test creation of onehot encoded batches."""
        self.dataset.build_token_vocab(self.train)
        self.dataset.encode(self.train, True)
        b = Batch(32, self.train)
        b.create_batches()

        expected = [True for batch in b]
        output = [all('encoded' in batch_item.__dict__ for batch_item in batch) for batch in b]
        self.assertEqual(output, expected, msg = "Not all items have an encoded element.")

        expected = [len(self.dataset.stoi) for batch in b for batch_item in batch]
        output = [getattr(batch_item, 'encoded').size(2) for batch in b for batch_item in batch]
        self.assertEqual(output, expected, msg = "Not all encoded items have the right length.")

    @unittest.skip("Not Implemented.")
    def test_encoded_batches(self):
        """Test creation of encoded batches. Not Implemented."""
        self.dataset.build_token_vocab(self.train)
        b = Batch(32, self.train)

        self.dataset.encode(self.train, False)
        expected = [True for batch in b]
        output = [all('encoded' in batch_item.__dict__ for batch_item in batch) for batch in b]
        self.assertEqual(output, expected, msg = "Not all items have an encoded element.")

        expected = [len(batch_item.text) for batch in b for batch_item in batch]
        output = [getattr(batch_item, 'encoded').size(2) for batch in b for batch_item in batch]
        self.assertEqual(output, expected, msg = "Not all encoded items have the right length.")


class TestBatchGenerator(unittest.TestCase):
    """Test the batchgenerator class."""

    @classmethod
    def setUpClass(cls):
        """Set up necessary class variables."""
        text_field = Field('text', train = True, label = False, ignore = False, ix = 6, cname = 'text')
        label_field = Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 5)
        ignore_field = Field('ignore', train = False, label = False, cname = 'ignore', ignore = True)

        fields = [ignore_field, ignore_field, ignore_field, ignore_field, ignore_field, label_field, text_field]

        cls.dataset = GeneralDataset(data_dir = '~/PhD/projects/active/Generalisable_abuse/data/',
                                     ftype = 'csv', fields = fields, train = 'davidson_test.csv', dev = None,
                                     test = None, train_labels = None, tokenizer = lambda x: x.split(),
                                     preprocessor = None, transformations = None,
                                     label_processor = None, sep = ',')
        cls.dataset.load('train')
        cls.train = cls.dataset.data
        cls.dataset.build_token_vocab(cls.train)
        cls.dataset.build_label_vocab(cls.train)
        cls.dataset.process_labels(cls.train)
        cls.dataset.encode(cls.train, True)
        cls.batch_size = 64
        batches = Batch(cls.batch_size, cls.train)
        batches.create_batches()
        cls.batches = batches

    @classmethod
    def tearDownClass(cls):
        """Tear down class between each test."""
        cls.dataset = 0
        cls.batches = 0

    def test_batch_generation(self):
        """Test the batchgenerator can access the variables."""
        batches = BatchExtractor('encoded', 'label', self.batches)

        for batch in batches:
            self.assertEqual(batch[0].size(0), batch[1].size(0))

    def test_tensorisation(self):
        """Test that the batches are all tensorized."""
        batches = BatchExtractor('encoded', 'label', self.batches)

        for batch in batches:
            self.assertIsInstance(batch[0], torch.Tensor, msg = "The type of the data element is incorrect.")
            self.assertIsInstance(batch[1], torch.Tensor, msg = "The type of the label element is incorrect.")

    def test_batch_sizes(self):
        """Test the __len__ function of a BatchExtractor."""
        batches = BatchExtractor('encoded', 'label', self.batches)
        self.assertEqual(len(batches), 15, msg = "The len operation on BatchExtractor is incorrect.")
