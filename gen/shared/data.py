import os
import csv
import json
import torch
import numpy as np
from tqdm import tqdm
from math import floor
from . import base
from collections import Counter, defaultdict
from torch.utils.data import IterableDataset


class GeneralDataset(IterableDataset):
    """A general dataset class, which loads a dataset, creates a vocabulary, pads, tensorizes, etc."""
    def __init__(self, data_dir: str, ftype: str, fields: base.FieldType, name: str, train: str,
                 dev: str = None, test: str = None, train_labels: str = None, dev_labels: str = None,
                 test_labels: str = None, sep: str = None, tokenizer: base.Union[base.Callable, str] = 'spacy',
                 preprocessor: base.Callable = None, transformations: base.Callable = None,
                 label_processor: base.Callable = None, label_preprocessor: base.Callable = None,
                 length: int = None, lower: bool = True) -> None:
        """Initialize the variables required for the dataset loading.
        :data_dir (str): Path of the directory containing the files.
        :ftype (str): ftype of the file ([C|T]SV and JSON accepted)
        :fields (base.List[base.Tuple[str, ...]]): Fields in the same order as they appear in the file.
                    Example: ('data', None)
        :name (str): Name of the dataset being used.
        :train (str): Path to training file.
        :dev (str, default None): Path to dev file, if dev file exists.
        :test (str, default = None): Path to test file, if test file exists.
        :train_labels (str, default = None): Path to file containing labels for training data.
        :dev_labels (str, default = None): Path to file containing labels for dev data.
        :test_labels (str, default = None): Path to file containing labels for test data.
        :sep (str, default = None): Separator token.
        :tokenizer (base.Callable or str, default = 'spacy'): Tokenizer to apply.
        :preprocessor (base.Callable, default = None): Preprocessing step to apply.
        :transformations (base.Callable, default = None): Method changing from one representation to another.
        :label_processor(base.Callable, default = None): Function to process labels with.
        :lower (bool, default = True): Lowercase the document.
        """
        self.data_dir = os.path.abspath(data_dir) if '~' not in data_dir else os.path.expanduser(data_dir)
        self.name = name
        super(GeneralDataset, self).__init__()

        try:
            ftype = ftype.upper()
            assert ftype in ['JSON', 'CSV', 'TSV']
            self.ftype = ftype
        except AssertionError as e:
            raise AssertionError("Input the correct file ftype: CSV/TSV or JSON")

        assert([getattr(f, 'label') is not None for f in fields])

        self.sep = sep
        self.fields = fields
        self.fields_dict = defaultdict(list)

        for field in self.fields:
            for key in field.__dict__:
                self.fields_dict[key].append(getattr(field, key))

        self.train_fields = [f for f in self.fields if f.train]
        self.label_fields = [f for f in self.fields if f.label]
        self.data_files = {key: os.path.join(self.data_dir, f) for f, key in zip([train, dev, test],
                                                                                 ['train', 'dev', 'test'])
                                                                                 if f is not None}
        self.label_files = {key: os.path.join(self.data_dir, f) for f, key in
                            zip([train_labels, dev_labels, test_labels], ['train', 'dev', 'test']) if f is not None}

        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.data_dir = data_dir
        self.repr_transform = transformations
        self.label_processor = label_processor if label_processor else self.label_name_lookup
        self.label_preprocessor = label_preprocessor
        self.length = length
        self.lower = lower

    def load(self, dataset: str = 'train', skip_header = True) -> None:
        """Load the datasebase.
        :skip_header (bool, default = True): Skip the header.
        :dataset (str, default = 'train'): Dataset to load. Must exist as key in self.data_files.
        """
        fp = open(self.data_files[dataset])

        if skip_header:
            next(fp)

        data = []
        for line in tqdm(self.reader(fp), desc = f'loading {self.name} ({dataset})'):
            data_line, datapoint = {}, base.Datapoint()  # TODO Look at moving all of this to the datapoint class.

            for field in self.train_fields:
                idx = field.index if self.ftype in ['CSV', 'TSV'] else field.cname
                data_line[field.name] = self.process_doc(line[idx].rstrip())
                data_line['original'] = self.process_doc(line[idx].rstrip())

            for field in self.label_fields:
                idx = field.index if self.ftype in ['CSV', 'TSV'] else field.cname
                if self.label_preprocessor:
                    data_line[field.name] = self.label_preprocessor(line[idx].rstrip())
                else:
                    data_line[field.name] = line[idx].rstrip()

            for key, val in data_line.items():
                setattr(datapoint, key, val)
            data.append(datapoint)
        fp.close()

        if self.length is None:
            # Get the max length
            lens = []
            for doc in data:
                for f in self.train_fields:
                    lens.append(len([tok for tok in getattr(doc, getattr(f, 'name'))]))
            self.length = max(lens)

        data = self.pad(data, self.length)

        if dataset == 'train':
            self.data = data
        elif dataset == 'dev':
            self.dev = data
        elif dataset == 'test':
            self.test = data

    def load_labels(self, dataset: str, label_name: str, label_path: str = None, ftype: str = None, sep: str = None,
                    skip_header: bool = True, label_processor: base.Callable = None,
                    label_ix: base.Union[int, str] = None) -> None:
        """Load labels from external file.
        :path (str): Path to data files.
        :dataset (str): dataset labels belong to.
        :label_file (str): Filename of data file.
        :ftype (str, default = 'CSV'): Filetype of the file.
        :sep (str, optional): Separator to be used with T/CSV files.
        :skip_header (bool): Skip the header.
        :label_processor: Function to process labels.
        :label_ix (int, str): Index or name of column containing labels.
        :label_name (str): Name of the label column/field.
        """
        path = label_path if label_path is not None else self.path
        ftype = ftype if ftype is not None else self.ftype
        sep = sep if sep is not None else self.sep

        labels = []
        fp = open(path)
        if skip_header:
            next(fp)

        if dataset == 'train':
            data = self.data
        elif dataset == 'dev':
            data = self.dev
        elif dataset == 'test':
            data = self.test

        labels = [line[label_ix.rstrip()] for line in self.reader(fp, ftype, sep)]

        for l, doc in zip(labels, data):
            setattr(doc, label_name, l)

    def set_labels(self, data, labels):
        for doc, label in zip(data, labels):
            setattr(doc, 'label', label)

    @property
    def train_set(self) -> base.DataType:
        """Set or get the training set."""
        return self.data

    @train_set.setter
    def train_set(self, train: base.DataType) -> None:
        self.data = train

    @property
    def dev_set(self) -> base.DataType:
        """Set or get the development set."""
        return self.data

    @dev_set.setter
    def train_set(self, dev: base.DataType) -> None:
        self.dev = dev

    @property
    def test_set(self) -> base.DataType:
        """Set or get the testelopment set."""
        return self.data

    @test_set.setter
    def train_set(self, test: base.DataType) -> None:
        self.test = test

    def reader(self, fp, ftype: str = None, sep: str = None):
        """Instatiate the reader to be used.
        :fp: Opened file.
        :ftype (str, default = None): Filetype if loading external data.
        :sep (str, default = None): Separator to be used.
        :return reader: Iterable objecbase.
        """
        ftype = ftype if ftype is not None else self.ftype
        if ftype in ['CSV', 'TSV']:
            sep = sep if sep else self.sep
            reader = csv.reader(fp, delimiter = sep)
        else:
            reader = self.json_reader(fp)
        return reader

    def json_reader(self, fp: str) -> base.Generator:
        """Create a JSON reading objecbase.
        :fp (str): Opened file objecbase.
        :return: """
        for line in fp:
            yield json.loads(line)

    def build_token_vocab(self, data: base.DataType, original: bool = True):
        """Build vocab over datasebase.
        :data (base.DataType): List of datapoints to process.
        :original (bool): Use the original document to generate vocab.
        """
        train_fields = self.train_fields
        self.token_counts = Counter()

        for doc in tqdm(data, desc = "Building vocabulary"):
            if original:
                self.token_counts.update(doc.original)
            else:
                for f in train_fields:
                    self.token_counts.update(getattr(doc, getattr(f, 'name')))

        self.token_counts.update({'<unk>': int(np.mean(list(self.token_counts.values())))})
        self.token_counts.update({'<pad>': int(np.mean(list(self.token_counts.values())))})

        self.itos = {ix: tok for ix, (tok, _) in enumerate(self.token_counts.most_common())}
        self.stoi = {tok: ix for ix, tok in self.itos.items()}

    def extend_vocab(self, data: base.DataType):
        """Extend the vocabulary.
        :data (base.DataType): List of datapoints to process.
        """
        for doc in data:
            start_ix = len(self.itos)
            for f in self.train_fields:
                tokens = getattr(doc, getattr(f, 'name'))
                self.token_counts.update(tokens)
                self.itos.update({start_ix + ix: tok for ix, tok in enumerate(tokens) if tok not in self.stoi.values()})

        self.stoi = {tok: ix for ix, tok in self.itos.items()}

    def limit_vocab(self, limiter: base.Callable, **kwargs) -> None:
        """Limit vocabulary using a function that returns a new vocabulary.
        :limiter (base.Callable): Function to limit the vocabulary.
        :kwargs: All arguments needed for the limiter function.
        """
        self.itos = limiter(self.token_counts, **kwargs)
        self.itos[len(self.itos)] = '<pad>'
        self.itos[len(self.itos)] = '<unk>'
        self.stoi = {tok: ix for ix, tok in self.itos.items()}

    def vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.itos)

    def vocab_token_lookup(self, tok: str) -> int:
        """Lookup a single token in the vocabulary.
        :tok (str): Token to look up.
        :return ix (int): Return the index of the vocabulary item.
        """
        try:
            ix = self.stoi[tok]
        except IndexError as e:
            ix = self.stoi['<unk>']
        return ix

    def vocab_ix_lookup(self, ix: int) -> str:
        """Lookup a single index in the vocabulary.
        :ix (int): Index to look up.
        :return tok (str): Returns token
        """
        return self.itos[ix]

    def build_label_vocab(self, labels: base.DataType) -> None:
        """Build label vocabulary.
        :labels (base.DataType): List of datapoints to process.
        """
        labels = set(getattr(l, getattr(f, 'name')) for l in labels for f in self.label_fields)
        self.itol = {ix: l for ix, l in enumerate(sorted(labels))}
        self.ltoi = {l: ix for ix, l in self.itol.items()}

    def label_name_lookup(self, label: str) -> int:
        """Look up label index from label.
        :label (str): Label to process.
        :returns (int): Return index value of label."""
        return self.ltoi[label]

    def label_ix_lookup(self, label: int) -> str:
        """Look up label index from label.
        :label (int): Label index to process.
        :returns (str): Return label."""
        return self.itol[label]

    def label_count(self) -> int:
        """Get the number of the labels."""
        return len(self.itol)

    def process_labels(self, data: base.DataType, processor: base.Callable = None):
        """Take a dataset of labels and process them.
        :data (base.DataType): Dataset of datapoints to process.
        :processor (base.Callable, optional): Custom processor to use.
        """
        for doc in data:
            label = self._process_label([getattr(doc, getattr(f, 'name')) for f in self.label_fields], processor)
            setattr(doc, 'label', label)

    def _process_label(self, label, processor: base.Callable = None) -> int:
        """Modify label using external function to process ibase.
        :label: Label to process.
        :processor: Function to process the label."""
        if not isinstance(label, list):
            label = [label]
        processor = processor if processor is not None else self.label_processor
        return [processor(l) for l in label]

    def process_doc(self, doc: base.DocType) -> list:
        """Process a single documenbase.
        :doc (base.DocType): Document to be processed.
        :return doc (list): Return processed doc in tokenized list formabase."""
        if isinstance(doc, list):
            doc = " ".join(doc)

        doc = doc.lower() if self.lower else doc

        doc = self.tokenizer(doc.replace("\n", " "))

        if self.preprocessor is not None:
            doc = self.preprocessor(doc)

        if self.repr_transform is not None:
            doc = self.repr_transform(doc)

        return doc

    def pad(self, data: base.DataType, length: int = None) -> list:
        """Pad each document in the datasets in the dataset or trim documenbase.
        :data (base.DataType): List of datapoints to process.
        :length (int, optional): The sequence length to be applied.
        :return doc: Return list of padded datapoints."""

        if not self.length and length is not None:
            self.length = length
        elif not self.length and length is None:
            raise AttributeError("A length must be given to pad tokens.")

        padded = []
        for doc in data:
            for field in self.train_fields:
                text = getattr(doc, getattr(field, 'name'))
                setattr(doc, getattr(field, 'name'), self._pad_doc(text, length))
                padded.append(doc)
        return padded

    def _pad_doc(self, text, length):
        """Do the actual padding.
        :text: The extracted text to be padded or trimmed.
        :length: The length of the sequence length to be applied.
        :return padded: Return padded document as a lisbase.
        """
        delta = length - len(text)
        padded = text[:delta] if delta < 0 else text + ['<pad>'] * delta
        return padded

    def encode(self, data: base.DataType, onehot: bool = True):
        """Encode a documenbase.
        :data (base.DataType): List of datapoints to be encoded.
        :onehot (bool, default = True): Set to true to onehot encode the documenbase.
        """
        # TODO Names need to be the same for all datasets used.
        names = [getattr(f, 'name') for f in self.train_fields]
        encoding_func = self.onehot_encode_doc if onehot else self.encode_doc
        for doc in tqdm(data, desc = "Encoding data"):
            encoded = encoding_func(doc, names)
            setattr(doc, 'encoded', encoded)
        return data

    def onehot_encode_doc(self, doc, names):
        """Onehot encode a single documenbase."""
        text = [tok for name in names for tok in getattr(doc, name)]
        encoded_doc = torch.zeros(1, self.length, len(self.stoi))

        if len(text) < self.length:  # For externally loaded datasets
            text = self._pad_doc(text, self.length)

        for ix in range(self.length):
            try:
                tok_ix = self.stoi['<unk>'] if text[ix] not in self.stoi else self.stoi[text[ix]]
            except IndexError:
                __import__('pdb').set_trace()
            encoded_doc[0][ix][tok_ix] = 1
        setattr(doc, 'encoded', encoded_doc)

        return encoded_doc

    def encode_doc(self, doc, names):
        """Encode documents using just the index of the tokens that are present in the documenbase."""

        raise NotImplementedError
        text = [tok for name in names for tok in getattr(doc, name)]
        length = sum(len(getattr(doc, name)) for name in names)
        encoded_doc = torch.LongTensor(1, self.length, length)  # batch, seq, doc length

        # 5 x 1 x [2, 35, 4, 0, 0] (last dimension is: indices in the document up to the document length)
        # 5 x 1 x len(vocab) (last dimension is: vocab length for each token)

        # ISSUE
        # We need to create a tensor containing a onehot tensor of each word.
        # CURRENT STATUS
        # A single elmeent containing the index of the current token.
        # GOAL
        # For each position, ensure it's the token for that position

        # Here we only have a tensor of a single token in the sentence. What we want is a onehot tensor
        for ix in range(self.length):
            tok_ix = self.stoi['<unk>'] if text[ix] not in self.stoi else self.stoi[text[ix]]
            encoded_doc[0][ix][ix] = tok_ix

        setattr(doc, 'encoded', encoded_doc)
        return encoded_doc

    def stratify(self, data, strata_field):
        # TODO Rewrite this code to make sense with this implementation.
        # TODO This doesn't make sense to me.
        strata_maps = defaultdict(list)
        for doc in data:
            strata_maps[getattr(doc, strata_field)].append(doc)
        return list(strata_maps.values())

    def split(self, data: base.DataType, splits: base.Union[int, base.List[int]],
              stratify: str = None) -> base.Tuple[base.DataType]:
        """Split the datasebase.
        :data (base.DataType): Dataset to splibase.
        :splits (int | base.List[int]]): Real valued splits.
        :stratify (str): The field to stratify the data along.
        :return data: Return splitted data.
        """
        if stratify is not None:
            data = self.stratify(data, )

        if isinstance(splits, float):
            splits = [splits]

        num_splits = len(splits)
        num_datapoints = len(data)
        splits = list(map(lambda x: floor(num_datapoints * x), splits))

        for ix, split in enumerate(splits):
            if split == 0:
                if 1 < splits[ix - 1] and ix + 1 != len(splits):
                    splits[ix] = splits[ix - 1] + 1
                else:
                    splits[ix] = 1

        if num_splits == 1:
            self.data = data[:splits[0]]
            self.test = data[splits[0]:]
            out = (self.data, self.test)
        elif num_splits == 2:
            self.data = data[:splits[0]]
            self.test = data[-splits[1]:]
            out = (self.data, self.test)
        elif num_splits == 3:
            self.data = data[:splits[0]]
            self.dev = data[splits[0]:splits[1]]
            self.test = data[-splits[2]:]
            out = (self.data, self.dev, self.test)
        return out

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        try:
            return len(self.data)
        except TypeError:
            return 2**32

    def __iter__(self):
        for x in self.data:
            yield x

    def __getattr__(self, attr):
        if attr in self.fields_dict:
            for x in self.data:
                yield getattr(x, attr)
