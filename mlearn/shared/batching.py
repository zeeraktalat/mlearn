import torch
from . import base


class TorchTextOnehotBatchGenerator:
    """A class to get the information from the batches."""

    def __init__(self, dataloader: base.DataType, datafield: str, labelfield: str, vocab_size: int):
        self.data, self.df, self.lf = dataloader, datafield, labelfield
        self.VOCAB_SIZE = vocab_size

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            X = torch.nn.functional.one_hot(getattr(batch, self.df), self.VOCAB_SIZE)
            y = getattr(batch, self.lf)
            yield (X, y)


class TorchTextDefaultExtractor:

    def __init__(self, datafield: str, labelfield: str, dataloader: base.DataType):
        self.data, self.df, self.lf = dataloader, datafield, labelfield

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            X = getattr(batch, self.df)
            y = getattr(batch, self.lf)
            yield (X, y)


class BatchExtractor:
    """A class to get the information from the batches."""

    def __init__(self, datafield: str, labelfield: str, batcher: base.DataType, dataset: base.DataType,
                 onehot: bool = True):
        self.batcher = batcher
        self.data = dataset
        self.onehot = onehot
        self.df, self.lf = datafield, labelfield

    def __len__(self):
        return len(self.batcher)

    def __iter__(self):
        for batch in self.batcher:
            batch = self.data.encode(batch, onehot = self.onehot)
            X = torch.cat([getattr(doc, self.df) for doc in batch], dim = 0)
            y = torch.tensor([getattr(doc, self.lf) for doc in batch]).flatten()
            yield (X, y)

    def __getitem__(self, i):
        return self.batcher[i]

    def shuffle(self):
        """Shuffle dataset."""
        self.batcher = self.batcher.shuffle()

    def shuffle_batches(self):
        """Shuffle the batch order."""
        self.batcher = self.batcher.shuffle_batches()


class Batch(object):
    """Create batches."""

    def __init__(self, batch_size: int, data: base.DataType):
        self.batch_size = batch_size
        self.data = data

    def create_batches(self, data = None):
        """Go over the data and create batches.
        :data (optional): Add a dataset to have batches created on."""

        data = self.data if not data else data

        self.batches, batch = [], []
        start_ix, end_ix = 0, self.batch_size

        for i in range(0, len(data), self.batch_size):
            batch = data[start_ix:end_ix]
            start_ix, end_ix = start_ix + self.batch_size, end_ix + self.batch_size
            self.batches.append(batch)

    def shuffle(self):
        data = [self.data[i] for i in torch.utils.data.RandomSampler(self.data)]
        self.create_batches(data)
        return self

    def shuffle_batches(self):
        self.batches = [self.batches[i] for i in torch.utils.data.RandomSampler(self.batches)]
        return self

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return self.batches[i]

    def __getattr__(self, item, attr):
        for doc in item:
            yield doc[attr]
