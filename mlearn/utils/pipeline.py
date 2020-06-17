from mlearn import base
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


def select_vectorizer(vectorizer: str) -> base.VectType:
    """
    Identify vectorizer used and return it to be used.

    :param vectorizer: Vectorizer to be used.
    :return v: Vectorizer function.
    """
    if not any(vec in vectorizer for vec in ['dict', 'count', 'hash', 'tfidf']):
        print("You need to select from the options: dict, count, hash, tfidf. Defaulting to Dict.")
        return DictVectorizer

    vect = vectorizer.lower()
    if 'dict' in vect:
        v = DictVectorizer
    elif 'tfidf' in vect:
        v = TfidfVectorizer
    elif 'hash' in vect:
        v = HashingVectorizer
    elif 'count' in vect:
        v = CountVectorizer
    setattr(v, 'fitted', False)

    return v


def vectorize(data: base.DataType, dataset: GeneralDataset, vect: base.VectType) -> base.DataType:
    """
    Vectorise documents.

    :dataset (data.GeneralDataset): Dataset object.
    :data (base.DataType): Dataset to vectorize.
    :vect (base.VectType): Vectorizer to use.
    :returns vectorized (base.DataType): Return vectorized dataset.
    """
    data = [getattr(doc, getattr(f, 'name')) for f in dataset.train_fields for doc in data]

    if vect.fitted:
        vectorized = vect.transform(data)
    else:
        vect.fit(data)
        vectorized = vect.transform(data)
        vect.fitted = True
    return vectorized


def top_features():
    raise NotImplementedError
