import torch
from mlearn import base


class TorchtextExtractor:
    """A class to extract information from batches."""

    def __init__(self, datafield, labelfield, dataname, data, vocab_size = None):
        self.data = data
        self.datafield = datafield
        self.labelfield = labelfield
        setattr(self.data, 'name', dataname)
        self.vocab_size = None

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            if self.vocab_size is not None:
                X = torch.nn.functional.one_hot(getattr(batch, self.datafield), self.vocab_size)
            else:
                X = getattr(batch, self.datafield)
            y = getattr(batch, self.labelfield)
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
        self.batches = None

    def create_batches(self, data: base.DataType = None):
        """
        Go over the data and create batches.

        :data (base.DataType, default = None): Add a dataset to have batches created on.
        """
        data = self.data if not data else data

        self.batches = []  # clear previously created batches (allows shuffling)
        for start_ix in range(0, len(data), self.batch_size):
            batch = data[start_ix:start_ix + self.batch_size]
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

    # def __getattr__(self, attr: str):
    #     """
    #     Get attribute from the batch.
    #
    #     :data (base.DataType): Dataset to get attributes from.
    #     :attr (str): Attribute to extract from the data.
    #     """
    #     for batch in self.batches:
    #         yield [getattr(doc, attr) for doc in batch]


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
        X = torch.cat([doc for doc in self.data.encode(self.batcher[idx], onehot = self.onehot)], dim = 0)
        y = torch.tensor([getattr(doc, self.lf) for doc in self.batcher[idx]]).flatten()
        return (X, y)

    def shuffle(self):
        """Shuffle dataset."""
        self.batcher = self.batcher.shuffle()
        return self

    def shuffle_batches(self):
        """Shuffle the batch order."""
        self.batcher = self.batcher.shuffle_batches()
        return self
