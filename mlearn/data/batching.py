import torch
from mlearn import base
from torch.nn.functional import one_hot
from . import Cython_Index


class TorchtextExtractor:
    """A class to extract information from batches."""

    def __init__(self, datafield, labelfield, dataname, data, vocab_size = None):
        self.data = data
        self.datafield = datafield
        self.labelfield = labelfield
        setattr(self.data, 'name', dataname)
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            if self.vocab_size is not None:
                X = one_hot(getattr(batch, self.datafield), self.vocab_size) # changed this since the import 
            else:
                X = getattr(batch, self.datafield)
            y = getattr(batch, self.labelfield)
            yield (X, y)


class Batch(base.Batch):
    """Create batches."""

    def __init__(self, batch_size: int, data: base.DataType, dataset: base.DataType, onehot: bool = True):
        """
        Initialize batch object.

        :batch_size (int): Size of batches.
        :data (base.DataType): The data to batch.
        """
        self.batch_size = batch_size
        self.data = data
        self.batches = None

        # from below 
        # self.batcher = batcher
        self.dataset = dataset
        self.onehot = onehot

        self.train_fields = dataset.train_fields
        self.length = dataset.length

        

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

    # Added these two methods below form extractor 

    def __getitem__(self, idx: int) -> base.DataType:
        """
        Get a batch given and index.

        :idx (int): Get an individual batch given and index.
        :returns (base.DataType): Batch.
        """
        X = torch.cat([doc for doc in self.encode(self.batcher[idx], onehot = self.onehot)], dim = 0) # self.data.encode
        y = torch.tensor([getattr(doc, self.lf) for doc in self.batcher[idx]]).flatten()
        return (X, y)

    # def __getattr__(self, attr: str):
    #     """
    #     Get attribute from the batch.
    #
    #     :data (base.DataType): Dataset to get attributes from.
    #     :attr (str): Attribute to extract from the data.
    #     """
    #     for batch in self.batches:
    #         yield [getattr(doc, attr) for doc in batch]


    def encode(self, data: base.DataType, onehot: bool = True) -> base.Iterator[base.DataType]:
        """
        Encode a documenbase.

        :data (base.DataType): List of datapoints to be encoded.
        :onehot (bool, default = True): Set to true to onehot encode the documenbase.
        :returns (base.Iterator[base.DataType]): Return documents encoded as tensors.
        """
        
        # for att in dir(data):
        #     print (att, getattr(data,att))

        encoding_func = self.onehot_encode_doc if onehot else self.index_encode_doc

        names = [getattr(f, 'name') for f in self.train_fields]



        for doc in data:
            text = [tok for name in names for tok in getattr(doc, name)]

            if len(text) != self.length:
                text = self._pad_doc(text, self.length)

            yield encoding_func(text, doc)

    def _pad_doc(self, text, length) -> base.List[str]: # copied this over to batching.py 
        """
        Do the actual padding.

        :text: The extracted text to be padded or trimmed.
        :length: The length of the sequence length to be applied.
        :return (base.List[str]): Return padded document as a list.
        """
        delta = length - len(text)
        padded = text[:delta] if delta < 0 else text + ['<pad>'] * delta
        return padded

    def onehot_encode_doc(self, text: base.DataType, doc: base.Datapoint) -> base.DataType:
        """
        Onehot encode a single document.

        :text (base.DataType): The document represented as a tokens.
        :doc (base.Datapoint): The datapoint to encode.
        :returns (base.DataType): Return onehot encoded tensor.
        """
        self.encoded = one_hot(self.encode_doc(text, doc).type(torch.long).unsqueeze(0) , len(self.dataset.stoi) ).type(torch.long).unsqueeze(0) #len(self.dataset.stoi)

                       #one_hot(getattr(batch, self.datafield), self.vocab_size)

        return self.encoded

    def index_encode_doc(self, text: base.DataType, doc: base.Datapoint) -> base.DataType:
        """
        Index encode a single document.

        :text (base.DataType): The document represented as a tokens.
        :doc (base.Datapoint): The datapoint to encode.
        :returns (base.DataType): Return onehot encoded tensor.
        """
        # This is Zeeraks code
        """self.encoded = self.encode_doc(text, doc).unsqueeze(0)"""
        # This is my Cython code 
        self.encoded = torch.from_numpy(Cython_Index.encode_doc_TEST(self.dataset,text)).unsqueeze(0)
        return self.encoded

    def encode_doc(self, text: base.DataType, doc: base.Datapoint) -> base.DataType:
        """
        Encode documents using just the index of the tokens that are present in the document.

        :text (base.DataType): The document represented as a tokens.
        :doc (base.Datapoint): The datapoint to encode.
        :returns (base.DataType): Return index encoded
        """
        # if hasattr(doc, 'encoded'):
        #     self.encoded = doc.encoded
        # else:
            
            # self.encoded = torch.tensor([dataset.stoi.get(text[ix], self.unk_tok) for ix in range(len(text))],
            #                        dtype = torch.long)
            
        self.encoded = torch.from_numpy(Cython_Index.encode_doc_TEST(self.dataset,text)) # came here
            
            # setattr(doc, 'encoded', self.encoded)
            
        return self.encoded


class BatchExtractor(Batch):
    """A class to get the information from the batches."""

    # currently dataset argument holds the stoi dict , we want to reference it in the general dataset class, when looking for vocab we want to reference the dataset object 
    # so self.stoi and self.itos need to exist in this dataset object
    # since we already have dataset as an arg in batchextractor we could just change the init from batch class to also make dataset an arg and then reference stoi from ther e

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
        super().__init__( len(self.batcher) , dataset , dataset) # telling code to remember that we are inherited from another obj, and then ini

        

    def __len__(self):
        """Get number of the batches."""
        return len(self.batcher)

    def shuffle(self):
        """Shuffle dataset."""
        self.batcher = self.batcher.shuffle()
        return self

    def shuffle_batches(self):
        """Shuffle the batch order."""
        self.batcher = self.batcher.shuffle_batches()
        return self

    def __iter__(self):
        """Iterate over batches in the data."""
        # Using global largeset batch file to encode, we want to encode per lenght of document 



        # look at longest file in the specific batch, not longest file in all files 
        # move all encoding to batch class 



        # move every method that has the word "encode" into the batch class straight up, 
        # TODO:CONSEQUENCE stoi.dict will not be available
        # modify batch class 
        for batch in self.batcher: # needs to use the cython code here, 
            X = torch.cat([doc for doc in self.encode( batch, onehot = self.onehot)], dim = 0)
            
            y = torch.tensor([getattr(doc, self.lf) for doc in batch]).flatten()
            yield (X, y) # come here from mtl_epoch
