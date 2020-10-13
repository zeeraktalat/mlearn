import numpy as np
from mlearn import base
from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score


class Metrics:
    """Metrics data object, to contain methods for computing, extracting, and evaluating different metrics."""

    def __init__(self, metrics: base.List[str], display_metric: str, early_stop: str = None, avg: str = 'macro'):
        """
        Intialize metrics computation class.

        :metrics (base.List[str]): List of strings containing metric names.
        :display_metric (str): Metric to display in TQDM loops.
        :early_stop (str, default = None): Metric to evaluate whether to perform early stopping.
        :avg (str, default = 'macro'): Averaging to use for metric functions.
        """
        self.scores, self.metrics = {}, OrderedDict()
        self.display_metric = display_metric
        self.early_stop = early_stop if early_stop is not None else display_metric
        self.average = avg

        self.select_metrics(metrics)  # Initialize the metrics dict.

    def select_metrics(self, metrics: base.List[str], loss: bool = True) -> None:
        """
        Select metrics for computation based on a list of metric names.

        :metrics (base.List[str]): List of metric names.
        :loss (bool, default = True): Keep track of loss as a metric.
        :return out: Dictionary containing name and methods.
        """
        if loss:
            metrics.append('loss')

        for m in metrics:
            m = m.lower()
            if 'accuracy' in m and 'accuracy':
                self.metrics['accuracy'] = accuracy_score
            elif 'precision' in m:
                self.metrics['precision'] = precision_score
            elif 'recall' in m:
                self.metrics['recall'] = recall_score
            elif 'auc' in m:
                self.metrics['auc'] = roc_auc_score
            elif 'confusion' in m:
                self.metrics['confusion'] = confusion_matrix
            elif 'f1' in m:
                self.metrics['f1-score'] = f1_score
                m = 'f1-score'
            self.scores[m] = []

    def compute(self, labels: base.DataType, preds: base.DataType, **kwargs) -> base.Dict[str, float]:
        """
        Compute scores for the model.

        :metrics (base.Dict[str, base.Callable]): Metrics dictionary.
        :labels (base.DataType): True labels.
        :preds (base.DataType): Predicted labels.
        :returns (base.Dict[str, float]): Dict containing computed scores.
        """
        for metric, score in self._compute(labels, preds, **kwargs).items():
            self.scores[metric].append(score)
        return self.scores

    def _compute(self, labels: base.DataType, preds: base.DataType, **kwargs) -> base.Dict[str, float]:
        """
        Compute scores for the model without storing them.

        :metrics (base.Dict[str, base.Callable]): Metrics dictionary.
        :labels (base.DataType): True labels.
        :preds (base.DataType): Predicted labels.
        :returns (base.Dict[str, float]): Dict containing computed scores.
        """
        scores = {}
        for name, metric in self.metrics.items():
            try:
                scores[name] = float(metric(preds, labels, average = self.average, zero_division = 0, **kwargs))
            except TypeError:
                scores[name] = float(metric(preds, labels, **kwargs))
        return scores

    @property
    def loss(self):
        """Add or get mean loss."""
        return np.mean(self.scores['loss'])

    @loss.setter
    def loss(self, value: float) -> None:
        """
        Add latest loss value to scores.

        :value (float): Loss value of latest run.
        """
        self.scores['loss'].append(value)

    def get_last(self, key: str) -> float:
        """
        Get last value for given key.

        :key (str): Metric to get last value of.
        :returns (float): Return last computed value for the key.
        """
        return self.scores[key][-1]

    def last_round(self) -> base.Dict[str, float]:
        """
        Get the last value for every metric that is computed.

        :returns comptued (base.Dict[str, float]): Returns the most recently computed value for each metric.
        """
        computed = {self.scores[key][-1] for key in self.scores}
        return computed

    def display(self) -> base.Dict[str, float]:
        """
        Get display metric dict.

        :returns (base.Dict[str, float]): display metric dict.
        """
        # zero-pad the left so we have at least two scores, in case we currently only have one score
        num_current_scores = len(self.scores[self.display_metric])
        if num_current_scores == 0:
            prev_score, cur_score = 0.0, 0.0
        elif num_current_scores == 1:
            prev_score, cur_score = 0.0, self.scores[self.display_metric][-1]
        else:
            prev_score, cur_score = self.scores[self.display_metric][-2], self.scores[self.display_metric][-1]
        difference = cur_score - prev_score
        return {self.display_metric: np.mean(self.scores[self.display_metric]), 'diff': difference}

    def last_display(self) -> float:
        """Get last display score."""
        return self.scores[self.display_metric][-1]

    def early_stopping(self) -> float:
        """Provide early stopping metrics."""
        return self.scores[self.early_stop][-1]

    def list(self) -> base.List:
        """Return a list of all metrics."""
        return list(self.metrics.keys())

    def __len__(self) -> int:
        """Compute the number of entries input into each list."""
        losses = len(self.scores['loss'])
        return len(self.scores[self.display_metric]) if losses == 0 else losses

    def __getitem__(self, metric: str) -> list:
        """
        Get individual metric.

        :metric (str): Metric to get results for.
        :returns (list): Scores for desired metric.
        """
        return self.scores[metric]
