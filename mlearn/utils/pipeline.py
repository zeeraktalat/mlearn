import numpy as np
from mlearn import base
from collections import defaultdict, Counter
from mlearn.data.dataset import GeneralDataset
from mlearn.data.batching import Batch, BatchExtractor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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


def select_vectorizer(vectorizer: str) -> base.VectType:
    """
    Identify vectorizer used and return it to be used.

    :param vectorizer: Vectorizer to be used.
    :return v: Vectorizer function.
    """
    if not any(vec in vectorizer for vec in ['dict', 'count', 'tfidf']):
        print("You need to select from the options: dict, count, tfidf. Defaulting to Dict.")
        return DictVectorizer

    vect = vectorizer.lower()
    if 'dict' in vect:
        v = DictVectorizer()
    elif 'tfidf' in vect:
        v = TfidfVectorizer()
    elif 'count' in vect:
        v = CountVectorizer()
    setattr(v, 'fitted', False)

    return v
