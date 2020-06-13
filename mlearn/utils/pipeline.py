from mlearn import base
from functools import reduce
from mlearn.data.dataset import GeneralDataset
from mlearn.data.batching import Batch, BatchExtractor


def process_and_batch(dataset: GeneralDataset, data: base.DataType, batch_size: int, onehot: bool = True):
    """
    Process a dataset and data.

    :dataset (GeneralDataset): The dataset object to use for processing.
    :data (base.DataType): The data to be batched and processed.
    :batch_size (int): Size of batches to create.
    :returns: Batched data.
    """
    # Process labels and encode data.
    dataset.process_labels(data)

    # Batch data
    batch = Batch(batch_size, data)
    batch.create_batches()
    batches = BatchExtractor('label', batch, dataset, onehot)
    return batches

def get_deep_dict_value(source: dict, keys: str, default = None):
    """
    Get values from deeply nested dicts.

    :source (dict): Dictionary to get data from.
    :keys (str): Keys split by '|'. E.g. outerkey|middlekey|innerkey.
    :default: Default return value.
    """"
    value = reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("|"), dictionary)
    return value
