import torch
import spacy
import numpy as np
from torch.nn import Module
from sklearn.base import ClassifierMixin, TransformerMixin
from typing import Callable, Dict, Iterator, List, Tuple, Union


class Field(object):
    """A class to set different properties of the individual fields."""

    def __init__(self, name: str, train: bool = False, label: bool = False, ignore: bool = False, ix: int = None,
                 cname: str = None):
        """
        Initialize the field object. Each individual field is to hold information about that field only.

        Example Use:
            train_field = Field('text', train = True, label = False, ignore = False, ix = 0)

        :name (str): Name of the field.
        :train (bool, default = False): Use for training.
        :label (bool, default = False): Indicate if it is a label field.
        :ignore (bool, default = False): Indicate whether to ignore the information in this field.
        :ix (int, default = None): Index of the field in the splitted file. Only set this for [C|T]SV files.
        :cname (str, default = None): Name of the column (/field) in the file. Only set this for JSON objects.
        """
        self.name = name
        self.train = train
        self.cname = cname
        self.label = label
        self.ignore = ignore
        self.index = ix


class Datapoint(object):
    """A class for each datapoint to instantiated as an object, which can allow for getting and setting attributes."""

    pass


class Batch(object):
    """A base class for batching procedures."""

    pass


# Do direct naming to prevent erroring out on CI
Callable = Callable
Dict = Dict
Iterator = Iterator
List = List
Tuple = Tuple
Union = Union

# Data types
FieldType = Field
DataType = Union[list, np.ndarray, torch.LongTensor, List[Datapoint]]
DocType = Union[str, List[str], spacy.tokens.doc.Doc, DataType]
ModelType = Union[Module]

# Model/Vectorizer Type
ModelType = Union[ClassifierMixin, Module]
VectType = TransformerMixin
