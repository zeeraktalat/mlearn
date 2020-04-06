import torch
from tqdm import tqdm
from mlearn import base
from mlearn.modeling.metrics import compute


def predict_torch_model(model: base.modelType, iterator: base.DataType, loss_f: base.Callable,
                        **kwargs) -> base.Tuple[list, list]:
    """Predict using trained model.
    :model (base.ModelType): Trained model to be trained.
    :iterator (base.DataType): Batched dataset to predict on.
    :loss_f (base.Callable): Loss function.
    :returns (): eval_loss, None, scores, None
    """
    predicted, labels = [], []
    loss = []

    for X, y in tqdm(iterator, desc = "Evaluating model", leave = False):
        pred = model(X, **kwargs).cpu()
        li = loss(pred, y)
        loss.append(li.data.item())

        predicted.extend(pred)
        labels.extend(y)

    predicted = torch.argmax(predicted, dim = 1)  # Get highest predicted label
    predicted = torch.stack(predicted).data.numpy().reshape(-1)
    labels = torch.stack(labels).data.numpy().reshape(-1)

    return list(predicted), list(labels), torch.mean(torch.Tensor(loss)).item()


def eval_torch_model(model: base.ModelType, iterator: base.DataType, loss_f: base.Callable,
                     metrics: base.Dict[str, base.Callable], mtl: bool = False, task_id: int = None, **kwargs):
    """Evalute pytorch model.

    :model (base.ModelType): Trained model to be trained.
    :iterator (base.DataType): Batched dataset to predict on.
    :loss_f (base.Callable): Loss function.
    :metrics (base.Dict[str, base.Callable]): Metrics dictionary.
    :mtl (bool, default = False): Is it a Multi-task Learning problem?
    :task_id (int, default = None): Task ID for MTL problem.
    :returns: TODO
    """
    with torch.no_grad():
        model.eval()

        if mtl and task_id is not None:
            predicted, true, loss = predict_torch_model(model, iterator, task_id = task_id, **kwargs)
        else:
            predicted, true, loss = predict_torch_model(model, iterator, **kwargs)

        scores = compute(metrics, true, predicted)

    return loss, None, scores, None


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
