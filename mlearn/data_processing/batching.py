import torch
from mlearn import base


class TorchTextOnehotBatchGenerator:
    """A class to get onehot-tensor batches from torchtext data object."""

    def __init__(self, dataloader: base.DataType, datafield: str, labelfield: str, vocab_size: int) -> None:
        """
        Initialize onehot batch generator for torchtext.

        :dataloader (base.DataType): The object loading the data.
        :datafield (str): Name of the field to access the data.
        :label_field (str): Name of the field to access the labels.
        :vocab_size (int): The size of the vocabulary.
        """
        self.data, self.df, self.lf = dataloader, datafield, labelfield
        self.VOCAB_SIZE = vocab_size

    def __len__(self) -> int:
        """Get length of the batches."""
        return len(self.data)

    def __iter__(self):
        """Iterate over batches in the data."""
        for batch in self.data:
            X = torch.nn.functional.one_hot(getattr(batch, self.df), self.VOCAB_SIZE)
            y = getattr(batch, self.lf)
            yield (X, y)


class TorchTextDefaultExtractor:
    """A class to get index-tensor batches from torchtext data object."""

    def __init__(self, datafield: str, labelfield: str, dataloader: base.DataType):
        """Initialize batch generator for torchtext."""
        self.data, self.df, self.lf = dataloader, datafield, labelfield

    def __len__(self):
        """Get length of the batches."""
        return len(self.data)

    def __iter__(self):
        """Iterate over batches in the data."""
        for batch in self.data:
            X = getattr(batch, self.df)
            y = getattr(batch, self.lf)
            yield (X, y)


class Batch(base.Batch):
    """Create batches."""

    def __init__(self, batch_size: int, data: base.DataType):
        """
        Initialize batch object.

        :batch_size (int): Size of batches.
        :data (base.DataType): The data to batch.
        """
        self.batch_size = batch_size
        self.data = data

    def create_batches(self, data: base.DataType = None):
        """
        Go over the data and create batches.

        :data (base.DataType, default = None): Add a dataset to have batches created on.
        """
        data = self.data if not data else data

        self.batches, batch = [], []
        start_ix, end_ix = 0, self.batch_size

        for i in range(0, len(data), self.batch_size):
            batch = data[start_ix:end_ix]
            start_ix, end_ix = start_ix + self.batch_size, end_ix + self.batch_size
            self.batches.append(batch)

    def shuffle(self):
        """Shuffle the dataset."""
        data = [self.data[i] for i in torch.utils.data.RandomSampler(self.data)]
        self.create_batches(data)
        return self

    def shuffle_batches(self):
        """Shuffle batches not individual datapoints."""
        self.batches = [self.batches[i] for i in torch.utils.data.RandomSampler(self.batches)]
        return self

    def __iter__(self) -> base.DataType:
        """Iterate over batches."""
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        """Get the number of batches."""
        return len(self.batches)

    def __getitem__(self, idx: int):
        """
        Get a batch given and index.

        :idx (int): Get an individual batch given and index.
        :returns (base.DataType): Batch.
        """
        return self.batches[idx]

    def __getattr__(self, data: base.DataType, attr: str):
        """
        Get attribute from the batch.

        :data (base.DataType): Dataset to get attributes from.
        :attr (str): Attribute to extract from the data.
        """
        for doc in data:
            yield doc[attr]


class BatchExtractor(base.Batch):
    """A class to get the information from the batches."""

    def __init__(self, labelfield: str, batcher: Batch, dataset: base.DataType, onehot: bool = True) -> None:
        """
        Initialize batch generator for the GeneralDataset.

        :labelfield (str): Name of the field to access the labels.
        :batcher (base.DataType): The Batch object to load the dataset.
        :dataset (base.DataType): The dataset to encode based on.
        :onehot (bool, default = True): Onehot encode data.
        """
        self.batcher = batcher
        self.data = dataset
        self.onehot = onehot
        self.lf = labelfield

    def __len__(self):
        """Get number of the batches."""
        return len(self.batcher)

    def __iter__(self):
        """Iterate over batches in the data."""
        for batch in self.batcher:
            X = torch.cat([doc for doc in self.data.encode(batch, onehot = self.onehot)], dim = 0)
            y = torch.tensor([getattr(doc, self.lf) for doc in batch]).flatten()
            yield (X, y)

    def __getitem__(self, idx: int) -> base.DataType:
        """
        Get a batch given and index.

        :idx (int): Get an individual batch given and index.
        :returns (base.DataType): Batch.
        """
        return self.batcher[idx]

    def shuffle(self):
        """Shuffle dataset."""
        self.batcher = self.batcher.shuffle()

    def shuffle_batches(self):
        """Shuffle the batch order."""
        self.batcher = self.batcher.shuffle_batches()
