from mlearn import base
from functools import reduce
from datetime import datetime
from mlearn.data.dataset import GeneralDataset
from mlearn.data.batching import Batch, BatchExtractor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def process_and_batch(dataset: GeneralDataset, data: base.DataType, batch_size: int, onehot: bool = True,
                      shuffle: bool = False, **kwargs):
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

    if shuffle:
        batches.shuffle()

    return batches


def get_deep_dict_value(source: dict, keys: str, default = None):
    """
    Get values from deeply nested dicts.

    :source (dict): Dictionary to get data from.
    :keys (str): Keys split by '|'. E.g. outerkey|middlekey|innerkey.
    :default: Default return value.
    """
    value = reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("|"), source)
    return value


def select_vectorizer(vectorizer: str = 'dict') -> base.VectType:
    """
    Identify vectorizer used and return it to be used.

    :vectorizer, default = 'dict': Vectorizer to be used.
    :return v: Vectorizer function.
    """
    vect = vectorizer.lower()
    if 'dict' in vect:
        v = DictVectorizer()
        setattr(v, 'name', 'DictVectorizer')
    elif 'tfidf' in vect:
        v = TfidfVectorizer()
        setattr(v, 'name', 'TFIDF-Vectorizer')
    elif 'count' in vect:
        v = CountVectorizer()
        setattr(v, 'name', 'CountVectorizer')
    setattr(v, 'fitted', False)

    return v


def _get_datestr():
    return datetime.now().strftime('%Y.%m.%d.%H.%M.%S')


def hyperparam_space(search_space: base.List[dict], hyper_parameters: base.List[base.Tuple]
                     ) -> base.List[dict]:
    """
    Create all hyper-parameter combinations to run.

    :search_space (base.List[dict]): List of dictionaries with one value
    :hyper_parameters (base.List[dict]): List of tuples containing a dict with all values for each iteration.
    :returns search_space (base.Generator[dict]): A list of dictionaries containing the search space.
    """
    for param_name, param_space in hyper_parameters:
        additions = []
        for comb_dict in search_space:
            for param in param_space:
                additions.append({**comb_dict, **{param_name: param}})
        search_space = additions
    return search_space
