import os
import csv
import json
import torch
import numpy as np
from tqdm import tqdm
from math import floor
from mlearn import base
from torch.nn.functional import one_hot
from collections import Counter, defaultdict
from torch.utils.data import Dataset as IterableDataset


class GeneralDataset(IterableDataset):
    """A general dataset class, which loads a dataset, creates a vocabulary, pads, tensorizes, etc."""

    def __init__(self, data_dir: str, ftype: str, fields: base.FieldType, name: str, train: str,
                 dev: str = None, test: str = None, train_labels: str = None, dev_labels: str = None,
                 test_labels: str = None, sep: str = None, tokenizer: base.Union[base.Callable, str] = 'spacy',
                 preprocessor: base.Callable = None, transformations: base.Callable = None,
                 label_processor: base.Callable = None, label_preprocessor: base.Callable = None,
                 length: int = None, lower: bool = True, gpu: bool = True) -> None:
        """
        Initialize the variables required for the dataset loading.

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
        :label_preprocessor(base.Callable, default = None): Function to preprocess labels.
        :lower (bool, default = True): Lowercase the document.
        :gpu (bool, default = True): Run on GPU.
        :length (int, default = None): Max length of documents.
        """
        self.data_dir = os.path.abspath(data_dir) if '~' not in data_dir else os.path.expanduser(data_dir)
        self.name = name
        super(GeneralDataset, self).__init__()

        try:
            ftype = ftype.upper()
            assert ftype in ['JSON', 'CSV', 'TSV']
            self.ftype = ftype
        except AssertionError:
            raise AssertionError("Input the correct file ftype: CSV/TSV or JSON")

        assert([getattr(f, 'label') is not None for f in fields])

        self.sep = sep
        self.fields = fields
        self.fields_dict = defaultdict(list)
        self.label_counts = defaultdict(int)

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
        self.gpu = gpu

    def load(self, dataset: str = 'train', skip_header = True, **kwargs) -> None:
        """
        Load the datasebase.

        :skip_header (bool, default = True): Skip the header.
        :dataset (str, default = 'train'): Dataset to load. Must exist as key in self.data_files.
        """
        fp = open(self.data_files[dataset])

        if skip_header:
            next(fp)

        data = []
        for line in tqdm(self.reader(fp), desc = f'Loading {self.name} ({dataset})'):

            data_line, datapoint = {}, base.Datapoint()  # TODO Look at moving all of this to the datapoint class.

            for field in self.train_fields:
                idx = field.index if self.ftype in ['CSV', 'TSV'] else field.cname
                data_line[field.name] = self.process_doc(line[idx].rstrip())
                data_line['original'] = line[idx].rstrip()

            for field in self.label_fields:
                idx = field.index if self.ftype in ['CSV', 'TSV'] else field.cname
                if self.label_preprocessor:
                    data_line[field.name] = self.label_preprocessor(line[idx].rstrip())
                else:
                    data_line[field.name] = line[idx].rstrip()

                self.label_counts[data_line[field.name]] += 1

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

        if dataset == 'train':
            self.data = data
        elif dataset == 'dev':
            self.dev = data
        elif dataset == 'test':
            self.test = data

    def load_labels(self, dataset: str, label_name: str, label_path: str = None, ftype: str = None, sep: str = None,
                    skip_header: bool = True, label_processor: base.Callable = None,
                    label_ix: base.Union[int, str] = None, **kwargs) -> None:
        """
        Load labels from external file.

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

    def set_labels(self, data: base.DataType, labels: base.DataType) -> None:
        """
        Set labels for documents.

        :data (base.DataType): Data to add label to.
        :labels (base.DataType): Labels to add to the labels.
        """
        for doc, label in zip(data, labels):
            setattr(doc, 'label', label)

    @property
    def train_set(self) -> base.DataType:
        """Set or get the training set."""
        return self.data

    @train_set.setter
    def train_set(self, train: base.DataType) -> None:
        """Set or get the training set."""
        self.data = train

    @property
    def dev_set(self) -> base.DataType:
        """Set or get the development set."""
        return self.dev

    @dev_set.setter
    def dev_set(self, dev: base.DataType) -> None:
        """Set or get the development set."""
        self.dev = dev

    @property
    def test_set(self) -> base.DataType:
        """Set or get the testelopment set."""
        return self.test

    @test_set.setter
    def test_set(self, test: base.DataType) -> None:
        """Set or get the testelopment set."""
        self.test = test

    @property
    def modify_length(self):
        """Get or set the max length of the documents."""
        return self.length

    @modify_length.setter
    def modify_length(self, x: int):
        """Get or set the max length of the documents."""
        self.length = x

    def reader(self, fp, ftype: str = None, sep: str = None) -> base.Callable:
        """
        Instatiate the reader to be used.

        :fp: Opened file.
        :ftype (str, default = None): Filetype if loading external data.
        :sep (str, default = None): Separator to be used.
        :return reader: Iterable object.
        """
        ftype = ftype if ftype is not None else self.ftype
        if ftype in ['CSV', 'TSV']:
            sep = sep if sep else self.sep
            reader = csv.reader(fp, delimiter = sep)
        else:
            reader = self.json_reader(fp)
        return reader

    def json_reader(self, fp: str) -> base.Iterator:
        """
        Create a JSON reading object.

        :fp (str): Opened file objecbase.
        :returns: Loaded lines.
        """
        for line in fp:
            yield json.loads(line)

    def build_token_vocab(self, data: base.DataType, original: bool = False) -> None:
        """
        Build vocab over datasebase.

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

        self.itos, self.stoi = {}, {}

        self.unk_tok = len(self.token_counts)
        self.pad_tok = len(self.token_counts) + 1

        try:
            del self.token_counts['<pad>']
        except KeyError:
            pass

        try:
            del self.token_counts['<unk>']
        except KeyError:
            pass

        self.stoi['<unk>'] = self.unk_tok
        self.stoi['<pad>'] = self.pad_tok
        self.itos[self.unk_tok] = '<unk>'
        self.itos[self.pad_tok] = '<pad>'
        for ix, (tok, _) in enumerate(tqdm(self.token_counts.most_common(), desc = "Encoding vocabulary")):
            self.itos[ix] = tok
            self.stoi[tok] = ix

    def extend_vocab(self, data: base.DataType) -> None:
        """
        Extend the vocabulary.

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
        """
        Limit vocabulary using a function that returns a new vocabulary.

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
        """
        Lookup a single token in the vocabulary.

        :tok (str): Token to look up.
        :return ix (int): Return the index of the vocabulary item.
        """
        try:
            ix = self.stoi[tok]
        except IndexError:
            ix = self.stoi['<unk>']
        return ix

    def vocab_ix_lookup(self, ix: int) -> str:
        """
        Lookup a single index in the vocabulary.

        :ix (int): Index to look up.
        :return tok (str): Returns token
        """
        return self.itos[ix]

    def build_label_vocab(self, labels: base.DataType) -> None:
        """
        Build label vocabulary.

        :labels (base.DataType): List of datapoints to process.
        """
        labels = set(getattr(l, getattr(f, 'name')) for l in labels for f in self.label_fields)
        self.itol, self.ltoi = {}, {}

        for ix, l in enumerate(tqdm(sorted(labels, reverse = True), desc = "Encode label vocab")):
            self.itol[ix] = l
            self.ltoi[l] = ix

    def label_name_lookup(self, label: str) -> int:
        """
        Look up label index from label.

        :label (str): Label to process.
        :returns (int): Return index value of label.
        """
        return self.ltoi[label]

    def label_ix_lookup(self, label: int) -> str:
        """
        Look up label index from label.

        :label (int): Label index to process.
        :returns (str): Return label.
        """
        return self.itol[label]

    def label_count(self) -> int:
        """Get the number of the labels."""
        return len(self.itol)

    def process_labels(self, data: base.DataType, processor: base.Callable = None):
        """
        Take a dataset of labels and process them.

        :data (base.DataType): Dataset of datapoints to process.
        :processor (base.Callable, optional): Custom processor to use.
        """
        for doc in data:
            label = self._process_label([getattr(doc, getattr(f, 'name')) for f in self.label_fields], processor)
            if isinstance(label, list):
                if len(label) > 1:
                    label = label
                else:
                    label = label[0]
            setattr(doc, 'label', label)
            # if len(label) > 1:
            #     setattr(doc, 'label', label)
            # elif isinstance(label, list):
            #     setattr(doc, 'label', label[0])

    def _process_label(self, label: base.List[str], processor: base.Callable = None) -> int:
        """
        Modify label using external function to process labels.

        :label (base.List[str]): Label to process.
        :processor (base.Callable): Function to process the label.
        :returns (int): Label processed as an int.
        """
        if not isinstance(label, list):
            label = [label]
        processor = processor if processor is not None else self.label_processor
        return [processor(l) for l in label]

    def process_doc(self, doc: base.DocType) -> list:
        """
        Process a single document.

        :doc (base.DocType): Document to be processed.
        :returns (list): Return processed doc in tokenized list format.
        """
        if isinstance(doc, list):
            doc = " ".join(doc)

        doc = doc.lower() if self.lower else doc

        doc = self.tokenizer(doc.replace("\n", " "))

        if self.preprocessor is not None:
            doc = self.preprocessor(doc)

        if self.repr_transform is not None:
            doc = self.repr_transform(doc)

        return doc

    def pad(self, data: base.DataType, length: int = None) -> base.List[base.Datapoint]:
        """
        Pad each document in the datasets in the dataset or trim documenbase.

        :data (base.DataType): List of datapoints to process.
        :length (int, optional): The sequence length to be applied.
        :returns (list): Return list of padded datapoints.
        """
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

    def _pad_doc(self, text, length) -> base.List[str]:
        """
        Do the actual padding.

        :text: The extracted text to be padded or trimmed.
        :length: The length of the sequence length to be applied.
        :return (base.List[str]): Return padded document as a list.
        """
        delta = length - len(text)
        padded = text[:delta] if delta < 0 else text + ['<pad>'] * delta
        return padded

    def encode(self, data: base.DataType, onehot: bool = True) -> base.Iterator[base.DataType]:
        """
        Encode a documenbase.

        :data (base.DataType): List of datapoints to be encoded.
        :onehot (bool, default = True): Set to true to onehot encode the documenbase.
        :returns (base.Iterator[base.DataType]): Return documents encoded as tensors.
        """
        names = [getattr(f, 'name') for f in self.train_fields]
        encoding_func = self.onehot_encode_doc if onehot else self.index_encode_doc

        for doc in data:
            text = [tok for name in names for tok in getattr(doc, name)]

            if len(text) != self.length:
                text = self._pad_doc(text, self.length)

            yield encoding_func(text, doc)

    def onehot_encode_doc(self, text: base.DataType, doc: base.Datapoint) -> base.DataType:
        """
        Onehot encode a single document.

        :text (base.DataType): The document represented as a tokens.
        :doc (base.Datapoint): The datapoint to encode.
        :returns (base.DataType): Return onehot encoded tensor.
        """
        encoded = one_hot(self.encode_doc(text, doc), len(self.stoi)).type(torch.long).unsqueeze(0)

        return encoded

    def index_encode_doc(self, text: base.DataType, doc: base.Datapoint) -> base.DataType:
        """
        Index encode a single document.

        :text (base.DataType): The document represented as a tokens.
        :doc (base.Datapoint): The datapoint to encode.
        :returns (base.DataType): Return onehot encoded tensor.
        """
        encoded = self.encode_doc(text, doc).unsqueeze(0)
        # encoded = encoded.unsqueeze(encoded.dim())

        return encoded

    def encode_doc(self, text: base.DataType, doc: base.Datapoint) -> base.DataType:
        """
        Encode documents using just the index of the tokens that are present in the document.

        :text (base.DataType): The document represented as a tokens.
        :doc (base.Datapoint): The datapoint to encode.
        :returns (base.DataType): Return index encoded
        """
        if hasattr(doc, 'encoded'):
            encoded = doc.encoded
        else:
            encoded = torch.tensor([self.stoi.get(text[ix], self.unk_tok) for ix in range(len(text))],
                                   dtype = torch.long)
            setattr(doc, 'encoded', encoded)
        return encoded

    def split(self, data: base.DataType = None, splits: base.List[float] = [0.8, 0.1, 0.1],
              store: bool = True, stratify: str = None, **kwargs) -> base.Tuple[base.DataType]:
        """
        Split the datasebase.

        :data (base.DataType, default = None): Dataset to split. If None, use self.data.
        :splits (int | base.List[int]], default = [0.8, 0.1, 0.1]): Size of each split.
        :store (bool, default = True): Store the splitted data in the object.
        :stratify (str): The field to stratify the data along.
        :return (base.Tuple[base.DataType]): Return splitted data.
        """
        data = self.data if data is None else data
        split_sizes = list(map(lambda x: floor(len(data) * x), splits))  # Get the actual sizes of the splits.

        if stratify is not None:  # TODO
            out = self._stratify_split(data, stratify, split_sizes, **kwargs)
        else:
            out = self._split(data, split_sizes)

        if store:
            self.data = out[0]
            self.test = out[-1]

            if len(splits) == 3:
                self.dev = out[1]
        return out

    def _stratify_split(self, data: base.DataType, strata_field: str, split_sizes: base.List[int], **kwargs
                        ) -> base.Tuple[list, base.Union[list, None], list]:
        """
        Stratify and split the data.

        :data (base.DataType): dataset to split.
        :split_sizes (int | base.List[int]): The number of documents in each split.
        :strata_field (str): Name of label field.
        :returns (base.Tuple[list, base.Union[list, None], list]): Return stratified splits.
        """
        train_size, dev_size, test_size = split_sizes[0], None, None
        idx_maps = defaultdict(list)

        # Create lists of each label.
        for i, doc in enumerate(data):
            idx_maps[getattr(doc, strata_field)].append(i)

        # Get labels and probabilities ordered
        labels, label_probs = zip(*{label: len(idx_maps[label]) / len(data) for label in idx_maps}.items())
        train = self._stratify_helper(data, labels, train_size, label_probs, idx_maps)

        num_splits = len(split_sizes)
        if num_splits == 1:
            test_size = len(data) - split_sizes[0]
        elif num_splits == 2:
            test_size = split_sizes[-1]
        elif num_splits == 3:
            dev_size = split_sizes[1]
            test_size = split_sizes[2]

        dev, test = None, None

        if dev_size is not None:
            dev = self._stratify_helper(data, labels, dev_size, label_probs, idx_maps)

        if test_size is None:
            if dev_size is not None:
                test_size = len(data) - (train_size + dev_size)
            else:
                test_size = len(data) - train_size
        else:
            if dev_size is not None:
                test_size += len(data) - (train_size + dev_size + test_size)
            else:
                test_size += len(data) - (train_size + test_size)

        indices = []
        for label in idx_maps:
            indices.extend(idx_maps[label])
        test = [data[ix] for ix in np.random.choice(indices, test_size, replace = False)]

        return train, dev, test

    def _stratify_helper(self, data: base.DataType, labels: tuple, sample_size: int, probs: tuple,
                         idx_map: dict) -> base.DataType:
        """
        Perform stratified allocation of documents.

        :data (base.DataType): Data to be split.
        :labels (tuple): The labels to choose from.
        :sample_size (int): Number of documents in the split.
        :probs (tuple): Probability for each label.
        :idx_map (dict): label to index (of documents with said label) map.
        :returns (base.DataType): Returns the stratified sample.
        """
        # Get counts for the label distribution
        label_count = Counter(np.random.choice(labels, sample_size, replace = True, p = probs))

        sampled = []
        for label, count in label_count.items():
            indices = np.random.choice(idx_map[label], count, replace = False)  # Get indices for label
            sampled.extend([data[ix] for ix in indices])
            idx_map[label] = [ix for ix in idx_map[label] if ix not in indices]  # Delete all used indices
        return sampled

    def _split(self, data: base.DataType, splits: base.Union[int, base.List[int]], **kwargs
               ) -> base.Tuple[list, base.Union[list, None], list]:
        """
        Split the dataset without stratification.

        :data (base.DataType): dataset to split.
        :splits (base.List[int]): The sizes of each split.
        :returns (base.Tuple[list, base.Union[list, None], list]): Tuple containing data splits.
        """
        indices = list(range(len(data)))
        num_splits = len(splits)
        if num_splits == 1:
            train, indices = self._split_helper(data, splits[0], indices)
            test, indices = self._split_helper(data, len(data) - splits[0], indices)
            out = (train, None, test)

        elif num_splits == 2:
            train, indices = self._split_helper(data, splits[0], indices)
            test, indices = self._split_helper(data, splits[1], indices)
            out = (train, None, test)

        elif num_splits == 3:
            train, indices = self._split_helper(data, splits[0], indices)
            dev, indices = self._split_helper(data, splits[1], indices)
            test, indices = self._split_helper(data, splits[2], indices)
            out = (train, dev, test)
        return out

    def _split_helper(self, data: base.DataType, size: int, indices: base.List[int]) -> base.Tuple[list, list]:
        """
        Allocate samples into the dataset.

        :data (base.DataType): Dataset to split.
        :size: (int): Size of the sample.
        :indices (base.List[int]): Indices for the entire dataset.
        :returns (base.Tuple[list, list]): Return the sampled data and unused indices.
        """
        sample = np.random.choice(indices, size, replace = False)
        sampled = [data[ix] for ix in sample]
        indices = [ix for ix in indices if ix not in sample]
        return sampled, indices

    def __getitem__(self, idx: int) -> base.Datapoint:
        """
        Get document in data given an index.

        :idx (int): Index of datapoint to extract.
        :returns (base.Datapoint): Datapoint in position idx.
        """
        return self.data[idx]

    def __len__(self):
        """Get the number of batches."""
        try:
            return len(self.data)
        except TypeError:
            return 2**32

    def __iter__(self):
        """Iterate over datapoints."""
        for x in self.data:
            yield x

    def __getattr__(self, attr):
        """
        Get attribute from the batch.

        :attr (str): Attribute to extract from the data.
        """
        if attr in self.fields_dict:
            for x in self.data:
                yield getattr(x, attr)