import torch
from tqdm import tqdm
from mlearn import base
from mlearn.utils import metrics


def predict_torch_model(model: base.ModelType, X, **kwargs) -> list:
    """
    Predict using trained model.

    :model (base.ModelType): Trained model to be trained.
    :X (base.DataType): Batch to predict on.
    :returns (list): Predictions.
    """
    return model(X, **kwargs).cpu()


def eval_torch_model(model: base.ModelType, iterator: base.DataType, loss_func: base.Callable, m: metrics.Metrics,
                     gpu: bool, mtl: int = None, task_id: int = None, store: bool = True, test: base.DataType = None,
                     **kwargs) -> None:
    """
    Evalute pytorch model.

    :model (base.ModelType): Trained model to be trained.
    :iterator (base.DataType): Batched dataset to predict on.
    :loss_func (base.Callable): Loss function.
    :m (object): Initialized m object.
    :gpu (bool): True if running on a GPU else false.
    :mtl (int, default = None): Task ID for MTL problem. Only unset if MTL model is in use.
    :store (bool, default = True): Store the prediction if true.
    :test (base.DataType, default = None): Data object to test on.
    :returns: None.
    """
    with torch.no_grad():
        model.eval()
        preds, labels = [], []
        loss = 0

        with tqdm(iterator, desc = "Evaluating model", leave = False) as loop:
            for X, y in loop:
                if gpu:
                    X = X.cuda()

                if mtl is not None:
                    if not isinstance(mtl, int):
                        raise AssertionError(f"AssertionError: MTL is not an INT. It has type: {type(mtl)}")
                    predicted = predict_torch_model(model, X, task_id = mtl)
                else:
                    predicted = predict_torch_model(model, X)

                loss += loss_func(predicted, y).data.item()
                preds.extend(torch.argmax(predicted, dim = 1).tolist())
                labels.extend(y.tolist())

            if store:
                for doc, pred in zip(test, predicted):
                    setattr(doc, 'pred', pred)

            m.compute(labels, preds)
            m.loss = loss / len(labels)


def predict_sklearn_model(model: base.ModelType, iterator: base.DataType, metrics: metrics.Metrics = None,
                          labels: base.DataType = None) -> base.Tuple[base.DataType, metrics.Metrics]:
    """
    Predict using trained Scikit-learn model.

    :model (base.ModelType): Trained model to be trained.
    :iterator (base.DataType): Dataset to predict on.
    :metrics (metrics.Metrics, default = None): Initialized Metrics object.
    :labels (base.DataType, default = None): For applying hte data
    :returns (Metrics.metrics): Metrics
    """
    preds = model.predict(iterator)
    if labels:
        metrics.compute(labels, preds)
    return preds, metrics


def eval_sklearn_model(model: base.ModelType, iterator: base.DataType, metrics: metrics.Metrics, labels: base.DataType,
                       store: bool = True, evalset: base.DataType = None):
    """
    Evaluate Scikit-learn model.

    :model (base.ModelType): Trained model to be trained.
    :iterator (base.DataType): Dataset to predict on.
    :metrics (object): Initialized Metrics object.
    :evalset (base.DataType): Data object being predicted on.
    :store (bool, default = True): Store the prediction if true.
    :returns (metrics.Metrics): Return evaluation metrics.
    """
    preds, metrics = predict_sklearn_model(model, iterator, metrics, labels)
    if store:
        for doc, lab, pred in zip(evalset, labels, preds):
            setattr(doc, 'pred', pred)


""" Joachim's Code, including regression evaluation.


def eval_model(model, X, y_true, task_id=0, batch_size=64):
    if model.binary:
        return eval_model_binary(model, X, y_true, task_id=task_id,
                                 batch_size=batch_size)
    else:
        return eval_model_regression(model, X, y_true, task_id=task_id,
                                     batch_size=batch_size)


def eval_model_regression(model, X, y_true, task_id=0, batch_size=64):
    predicted = predict_model(model, X, task_id, batch_size)
    mae, rank_corr = 0, float('nan')
    mae = mean_absolute_error(y_true, predicted)
    if predicted.sum() > 0:
        rank_corr = spearmanr(y_true, predicted)[0]
    return mae, rank_corr, predicted


def eval_model_binary(model, X, y_true, task_id=0, batch_size=64):
    predicted = predict_model(model, X, task_id, batch_size)
    f1 = f1_score(y_true, predicted)
    if predicted.sum() > 0:
        rank_corr = spearmanr(y_true, predicted)[0]
    else:
        rank_corr = float('nan')
    return f1, rank_corr, predicted
"""
