from . import base
from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix


def select_metrics(metrics: base.List[str]) -> base.Dict[str, base.Callable]:
    """Select metrics for computation based on a list of metric names.
    :metrics: List of metric names.
    :return out: Dictionary containing name and methods.
    """
    out = OrderedDict()
    if not isinstance(metrics, list):
        metrics = [metrics]

    for m in metrics:
        m = m.lower()
        if 'accuracy' in m and 'accuracy' not in out:
            out['accuracy'] = accuracy_score
        elif 'precision' in m and 'precision' not in out:
            out['precision'] = precision_score
        elif 'recall' in m and 'recall' not in out:
            out['recall'] = recall_score
        elif 'auc' in m and 'auc' not in out:
            out['auc'] = roc_auc_score
        elif 'confusion' in m and 'confusion' not in out:
            out['confusion'] = confusion_matrix

    return out
