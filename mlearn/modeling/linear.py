import os
import numpy as np
from mlearn import base
from collections import defaultdict, Counter
from mlearn.data.dataset import GeneralDataset
from mlearn.data.fileio import load_model, store_model


class LinearModel(object):

    def __init__(self, model: base.ModelType, model_name: str, vectorizer: base.Callable, **kwargs) -> None:
        """
        Initialize Linear model.

        :model (base.ModelType): Untrained linear model.
        :model_name (str): Name of the model.
        :vectorizer (base.Callable): Vectorizer for the model.
        """
        self.model = model(**kwargs)
        self.name = model_name
        self.vect = vectorizer
        self.library = 'sklearn'

    def top_features(self, dataset: GeneralDataset) -> dict:
        """
        Identify top features for scikit-learn model.

        :model (base.ModelType): Trained model to identify features for.
        :dataset (GeneralDataset): Dataset holding the label information.
        """
        coefs = defaultdict(Counter)
        ix2feat = {ix: feat for feat, ix in self.vect.vocabulary_.items()}

        for i, c in enumerate(range(dataset.label_count())):
            if i == 1 and dataset.label_count() == 2:
                break  # Task is binary so only class dimension in the feature matrices.

            if 'RandomForest' in self.name:
                update = {ix2feat[f]: self.model.feature_importances_[f] for f in
                          np.argsort(self.model.feature_importances_)}
            elif 'SVM' in self.name:
                update = {ix2feat[v]: self.model.coef_[i, v] for v in range(self.model.coef_.shape[1])}
            elif 'LogisticRegression' in self.name:
                update = {ix2feat[f]: self.model.coef_[i, f] for f in np.argsort(self.model.coef_[i])}

            coefs[i].update(update)
        return coefs

    def load_model(self, base_path: str):
        """
        Load model and vectorizer.

        :base_path (str): Base path to the model.
        """
        if os.path.isfile(f'{base_path}.mdl'):
            self.model, self.vect = load_model(base_path, library = 'sklearn')
        else:
            self.model, self.vect = load_model(f'{base_path}_{self.model.name}')

    def save_model(self, base_path: str):
        """
        Save model and vectorizer.

        :base_path (str): Base path to the model.
        """
        store_model(self, f'{base_path}_{self.name}', library = 'sklearn')
