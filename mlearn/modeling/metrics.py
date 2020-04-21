from mlearn import base
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score


class Metrics:
    """Metrics data object, to contain methods for computing, extracting, and evaluating different metrics."""

    def __init__(self, metrics: base.List[str], display_metric: str, early_stop: str):
        """
        Intialize metrics computation class.

        :metrics (base.List[str]): List of strings containing metric names.
        :display_metric (str): Metric to display in TQDM loops.
        :early_stop (str): Metric to evaluate whether to perform early stopping.
        """
        self.scores, self.metrics = {}, {}
        self.display = display_metric
        self.early_stop_metric = early_stop

        self.select_metrics(metrics)  # Initialize the metrics dict.

    def select_metrics(self, metrics: base.List[str]) -> None:
        """
        Select metrics for computation based on a list of metric names.

        :metrics: List of metric names.
        :return out: Dictionary containing name and methods.
        """
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

    def compute(self, labels: base.DataType, preds: base.DataType, **kwargs) -> base.Dict[str, float]:
        """
        Compute scores for the model.

        :metrics (base.Dict[str, base.Callable]): Metrics dictionary.
        :labels (base.DataType): True labels.
        :preds (base.DataType): Predicted labels.
        :returns (base.Dict[str, float]): Dict containing computed scores.
        """
        self.scores = self._compute(labels, preds)
        return self.scores

    def _compute(self, labels: base.DataType, preds: base.DataType, **kwargs) -> base.Dict[str, float]:
        """
        Compute scores for the model without storing them.

        :metrics (base.Dict[str, base.Callable]): Metrics dictionary.
        :labels (base.DataType): True labels.
        :preds (base.DataType): Predicted labels.
        :returns (base.Dict[str, float]): Dict containing computed scores.
        """
        scores = {name: float(metric(preds, labels, **kwargs)) for name, metric in self.metrics.items()}
        return scores

    def display_metric(self) -> base.Dict[str, float]:
        """
        Get display metric dict.

        :returns (base.Dict[str, float]): display metric dict.
        """
        return {self.display: self.scores[self.display]}

    def early_stopping(self) -> float:
        """
        Get score for metric for identifying early stopping.

        :returns (float): Score for early stopping metric.
        """
        return self.scores[self.early_stop_metric]

    def __getitem__(self, metric: str) -> float:
        """
        Get individual metric.

        :metric (str): Metric to get results for.
        :returns (float): Score for desired metric.
        """
        return self.metrics[metric]
