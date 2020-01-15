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

    def __init__(self, datafield: str, labelfield: str, dataloader: base.DataType):
        self.data, self.df, self.lf = dataloader, datafield, labelfield

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            X = torch.cat([getattr(doc, self.df) for doc in batch], dim = 0)
            y = torch.tensor([getattr(doc, self.lf) for doc in batch]).flatten()
            yield (X, y)


class Batch(object):
    """Create batches."""

    def __init__(self, batch_size: int, data: base.DataType):
        self.batch_size = batch_size
        self.data = data

    def create_batches(self):
        """Go over the data and create batches."""
        self.batches = []
        batch = []
        start_ix, end_ix = 0, self.batch_size
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[start_ix:end_ix]
            start_ix, end_ix = start_ix + self.batch_size, end_ix + self.batch_size
            self.batches.append(batch)

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
