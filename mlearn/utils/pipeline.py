import numpy as np
from mlearn import base
from collections import defaultdict, Counter
from mlearn.data.dataset import GeneralDataset
from mlearn.data.batching import Batch, BatchExtractor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer


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


def top_sklearn_features(model: base.ModelType, dataset: GeneralDataset, vect: base.VectType):
    """
    Identify top features for scikit-learn model.

    :model (base.ModelType): Trained model to identify features for.
    :dataset (GeneralDataset): Dataset holding the label information.
    :vect (base.VectType): Fitted vectorizer.
    """
    if dataset.label_count == 2:
        coefs = binary_sklearn_features(model, dataset, vect)
    elif dataset.label_count > 2:
        coefs = multinomial_sklearn_features(model, dataset, vect)
    return coefs


def binary_sklearn_features(model: base.ModelType, dataset: GeneralDataset, vect: base.VectType) -> dict:
    """
    Identify top features for binary scikit-learn model.

    :model (base.ModelType): Trained model to identify features for.
    :dataset (GeneralDataset): Dataset holding the label information.
    :vect (base.VectType): Fitted vectorizer.
    :return coefs (dict): Returns coefficient dictionary.
    """
    coefs = defaultdict(Counter())

    if 'RandomForest' in model.name:
        coefs[0].update({vect.feature_names_[f]: model.feature_importances_[f]
                         for f in np.argsort(model.feature_importances_)})
    elif 'SVM' in model.name:
        coefs[0].update({vect.feature_names_[v]: model.coef_[0, v] for v in range(model.coef_.shape[1])})
    elif 'LogisticRegression' in model.name:
        coefs[0].update({vect.feature_names_[f]: model.coef_[0, f] for f in np.argsort(model.coef_[0])})
    return coefs


def multinomial_sklearn_features(model: base.ModelType, dataset: GeneralDataset, vect: base.VectType) -> dict:
    """
    Identify top features for multiclass scikit-learn model.

    :model (base.ModelType): Trained model to identify features for.
    :dataset (GeneralDataset): Dataset holding the label information.
    :vect (base.VectType): Fitted vectorizer.
    """
    raise NotImplementedError

    coefs = defaultdict(Counter)
    if 'RandomForest' in model.name:
        pass
    elif 'SVM' in model.name:
        pass
    elif 'LogisticRegression' in model.name:
        for i, c in enumerate(dataset.label_count()):
            coefs[i].update({vect.feature_names_[f]: model.coef_[i, f] for f in np.argsort(model.coef_[i])})

    return coefs
