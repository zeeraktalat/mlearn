import os
import torch
import unittest
import torchtestcase
from mlearn.base import Field, Datapoint
from mlearn.data.dataset import GeneralDataset


class TestDataSet(torchtestcase.TorchTestCase):

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
        cls.json_dataset = GeneralDataset(data_dir = os.getcwd() + '/tests/',
                                          ftype = 'json', fields = fields, train = 'train.json',
                                          dev = 'garcia_stormfront_test.tsv', test = 'test.json', train_labels = None,
                                          tokenizer = lambda x: x.split(),
                                          preprocessor = None, transformations = None,
                                          label_processor = lambda x: x, sep = ',', name = 'test',
                                          length = 200)
        cls.csv_dataset.load('test')
        cls.test = cls.csv_dataset.test

    @classmethod
    def tearDown(cls):
        """Take down class."""
        cls.csv_dataset = 0
        cls.json_dataset = 0

    def test_load(self):
        """Test dataset loading."""
        expected = [("me gusta comer en la cafeteria".lower().split(), "SPANISH"),
                    ("Give it to me".lower().split(), "ENGLISH"),
                    ("No creo que sea una buena idea".lower().split(), "SPANISH"),
                    ("No it is not a good idea to get lost at sea".lower().split(), "ENGLISH")]

        csv_train = self.train
        output = [(doc.text, doc.label) for doc in csv_train]
        self.assertListEqual(output, expected, msg = 'Data Loading failed.')

        self.json_dataset.load('train', skip_header = False)
        json_train = self.json_dataset.data
        output = [(doc.text, doc.label) for doc in json_train]
        self.assertListEqual(output, expected, msg = 'Data Loading failed.')
        self.assertIsInstance(json_train[0], Datapoint, msg = 'Data Loading failed gave wrong type.')

        fields = [Field('text', train = True, label = False, ignore = False, ix = 0, cname = 'text'),
                  Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 1)]

        with self.assertRaises(AssertionError):
            GeneralDataset(data_dir = os.getcwd() + '/tests/',
                           ftype = 'jsn', fields = fields, train = 'train.json', dev = None,
                           test = 'test.json', train_labels = None, tokenizer = lambda x: x.split(),
                           preprocessor = None, transformations = None,
                           label_processor = None, sep = ',', name = 'test')

    def test_build_token_vocab(self):
        """Test vocab building method."""
        expected = set(['<pad>', '<unk>'] + list(sorted("""me gusta comer en la cafeteria Give it to me
                   No creo que sea una buena idea No it is not a good idea to get lost at sea""".lower().split())))
        self.csv_dataset.build_token_vocab(self.train)
        output = set(sorted(self.csv_dataset.stoi.keys()))
        self.assertSetEqual(output, expected, msg = 'Vocab building failed.')

        # Use original as the data set.
        expected = set(['<pad>', '<unk>'] + list(sorted("""me gusta comer en la cafeteria Give it to me
                   No creo que sea una buena idea No it is not a good idea to get lost at sea""".split())))
        self.csv_dataset.build_token_vocab(self.train, original = True)
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
        expected  = ["Yo creo que si".lower().split(),
                     "it is lost on me".lower().split()]
        self.assertListEqual(test[0].text, expected[0])
        self.assertListEqual(test[1].text, expected[1])

    @torchtestcase.skip("Not implemented yet.")
    def test_load_labels_from_file(self):
        """Test loading of labels from a labelfile."""
        with self.assertRaises(NotImplementedError):
            self.csv_dataset.load_labels('test')

    def test_vocab_token_lookup(self):
        '''Test looking up in vocab.'''
        self.csv_dataset.build_token_vocab(self.train)
        expected = 0
        output = self.csv_dataset.vocab_token_lookup('me')
        self.assertEqual(output, expected, msg = 'Vocab token lookup failed.')

        self.assertEqual(self.csv_dataset.vocab_token_lookup('shaggy'), self.csv_dataset.unk_tok,
                         msg = "UNK token not returned when unknown word is enountered.")

    def test_vocab_ix_lookup(self):
        '''Test looking up in vocab.'''
        self.csv_dataset.build_token_vocab(self.train)
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
        expected = 0
        self.assertEqual(output, expected, msg = 'label name lookup failed.')

    def test_label_ix_lookup(self):
        '''Test looking up in label.'''
        self.csv_dataset.build_label_vocab(self.train)
        output = self.csv_dataset.label_ix_lookup(1)
        expected = 'ENGLISH'
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
        expected = [0]
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
        expected = torch.zeros(2, 12, 25, dtype = torch.int64)

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

        output = torch.cat([datapoint for datapoint in self.csv_dataset.encode(test, onehot = True)], dim = 0)
        self.assertEqual(output, expected, msg = 'Onehot encoding failed.')

    def test_index_encoding(self):
        """Test the encoding. Not Implemented."""
        self.csv_dataset.build_token_vocab(self.train)

        expected = [torch.LongTensor([23, 12, 13, 23, 24, 24, 24, 24, 24, 24, 24, 24]).unsqueeze(0),
                    torch.LongTensor([1, 16, 21, 23, 0, 24, 24, 24, 24, 24, 24, 24]).unsqueeze(0)]
        output = [datapoint for datapoint in self.csv_dataset.encode(self.test, onehot = False)]

        for i, (out, exp) in enumerate(zip(expected, output)):
            self.assertEqual(out, exp, msg = f'Index encoding failed for doc {i}')

    def test_split(self):
        """Test splitting functionality."""
        expected = [3, 1]  # Lengths of the respective splits
        train, _, test = self.csv_dataset.split(self.train, [0.75])
        output = [len(train), len(test)]
        self.assertEqual(sum(output), len(self.train), msg = "The splits ([0.75]) != len(self.train)")
        self.assertListEqual(expected, output, msg = 'Split with ratio [0.75] failed.')

        expected = [3, 1]
        train, _, test = self.csv_dataset.split(self.train, [0.75, 0.25])
        output = [len(train), len(test)]
        self.assertEqual(sum(output), len(self.train), msg = "The splits ([0.75, 0.25]) != len(self.train)")
        self.assertListEqual(expected, output, msg = 'Two split ([0.75, 0.25] values in list failed.')

        expected = [2, 1, 1]
        train, dev, test = self.csv_dataset.split(self.train, [0.5, 0.25, 0.25])
        output = [len(train), len(dev), len(test)]
        self.assertEqual(sum(output), len(self.train), msg = "The splits ([0.50, 0.25, 0.25]) != len(self.train)")
        self.assertListEqual(expected, output, msg = 'Three split ([0.50, 0.25, 0.25]) values in list failed.')

    def test_stratified_split(self):
        """Test stratified splitting."""
        fields = [Field('text', train = True, label = False, ignore = False, ix = 5, cname = 'text'),
                  Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 4)]

        data = GeneralDataset(data_dir = os.getcwd() + '/tests/',
                              ftype = 'csv', fields = fields, train = 'garcia_stormfront_train.tsv',
                              dev = None, test = None, train_labels = None,
                              tokenizer = lambda x: x.split(),
                              preprocessor = None, transformations = None,
                              label_processor = None, sep = '\t', name = 'test')
        data.load('train')
        loaded_train = data.data

        expected = [1531, 383]  # Lengths of the respective splits
        train, _, test = data.split(loaded_train, splits = [0.8], stratify = 'label', store = False)
        output = [len(train), len(test)]

        self.assertEqual(sum(output), len(loaded_train), msg = f"The splits ([0.8]) != len(self.train)")
        self.assertListEqual(expected, output, msg = 'Splitting with just float failed.')

        expected = [1531, 383]
        train, _, test = data.split(loaded_train, [0.8, 0.2], stratify = 'label', store = False)
        train, _, test = self.csv_dataset.split(loaded_train, [0.8, 0.2], stratify = 'label', store = False)
        output = [len(train), len(test)]

        self.assertEqual(sum(output), len(loaded_train), msg = f"The splits ([0.8, 0.2]) != len(loaded_train)")
        self.assertListEqual(expected, output, msg = 'Two split values in list failed.')

        expected = [1531, 191, 192]
        train, dev, test = data.split(loaded_train, [0.8, 0.1, 0.1], stratify = 'label', store = False)
        train, dev, test = self.csv_dataset.split(loaded_train, [0.8, 0.1, 0.1], stratify = 'label', store = False)
        output = [len(train), len(dev), len(test)]

        self.assertEqual(sum(output), len(loaded_train), msg = f"The splits ([0.8, 0.1, 0.1]) != len(loaded_train)")
        self.assertListEqual(expected, output, msg = 'Three split values in list failed.')

    def test_properties(self):
        """Test setters and getters."""
        # Test train
        train = self.csv_dataset.train_set
        self.assertListEqual(train, self.csv_dataset.data, msg = "train_set does not return train data.")
        train.extend([1])
        self.csv_datase.train_set = train
        self.assertListEqual(train, self.csv_dataset.data, msg = "train_set does not set training data.")

        # Test dev
        dev = self.csv_dataset.dev_set
        self.assertListEqual(dev, self.csv_dataset.data, msg = "dev_set does not return dev data.")
        dev.extend([1])
        self.csv_datase.dev_set = dev
        self.assertListEqual(dev, self.csv_dataset.data, msg = "dev_set does not set deving data.")

        # Test test
        test = self.csv_dataset.test_set
        self.assertListEqual(test, self.csv_dataset.data, msg = "test_set does not return test data.")
        test.extend([1])
        self.csv_datase.test_set = test
        self.assertListEqual(test, self.csv_dataset.data, msg = "test_set does not set testing data.")

        # Test document length
        length = self.csv_dataset.modify_length  # TODO Replace with hardcoded number.
        self.assertEqual(length, self.csv_dataset.length, msg = "Retrieving length failed.")
        length += 1
        self.modify_length = length
        self.assertEqual(length, self.csv_dataset.length, msg = "Modifying length failed.")
