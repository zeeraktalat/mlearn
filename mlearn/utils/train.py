import torch
import numpy as np
from mlearn import base
from tqdm import tqdm, trange
from mlearn.utils.evaluate import eval_torch_model
from mlearn.utils.pipeline import process_and_batch
from mlearn.utils.early_stopping import EarlyStopping
from mlearn.data.fileio import write_predictions, write_results


def run_singletask_model(library: str, train: bool, writer: base.Callable, model_info: list, head_len: int, **kwargs):
    """
    Train or evaluate model.

    :library (str): Library of the model.
    :train (bool): Whether it's a train or test run.
    :writer (csv.writer): File to output model performance to.
    :model_info (list): Information about the model to be added to each line of the output.
    :head_len (int): The length of the header.
    """
    if train:
        func = train_singletask_model if library == 'pytorch' else train_sklearn_model
    else:
        func = eval_torch_model if library == 'pytorch' else evaluate_sklearn_model

    train_loss, dev_loss, train_scores, dev_scores = func(**kwargs)
    write_results(writer, train_scores, train_loss, dev_scores, dev_loss, model_info = model_info, exp_len = head_len,
                  **kwargs)

    if not train:
        write_predictions(kwargs['test_obj'], model_info = model_info, **kwargs)


def _singletask_epoch(model: base.ModelType, optimizer: base.Callable, loss_func: base.Callable,
                      iterator: base.DataType, clip: float = None, gpu: bool = True, **kwargs):
    """
    Training procedure for single task pytorch models.

    :model (base.ModelType): Untrained model to be trained.
    :optimizer (bas.Callable): Optimizer function.
    :loss_func (base.Callable): Loss function to use.
    :iterator (base.DataType): Batched training set.
    :clip (float, default = None): Add gradient clipping to prevent exploding gradients.
    :gpu (bool, default = True): Run on GPU
    :returns: TODO
    """
    predictions, labels = [], []
    epoch_loss = []

    with tqdm(iterator, desc = "Batch", leave = False) as loop:

        for X, y in loop:
            if gpu:
                X = X.cuda()
                y = y.cuda()

            scores = model(X, **kwargs)

            loss = loss_func(scores, y)
            epoch_loss.append(float(loss.data.item()))

            loss.backward()

            if clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)

            optimizer.step()

            predictions.extend(torch.argmax(scores, 1).cpu().tolist())
            labels.extend(y.cpu().tolist())

            loop.set_postfix(batch_loss = f"{epoch_loss[-1] / len(y):.7f}", epoch_loss = f"{np.mean(epoch_loss)}")

    return predictions, labels, epoch_loss


def train_singletask_model(model: base.ModelType, save_path: str, epochs: int, iterator: base.DataType,
                           loss_func: base.Callable, optimizer: base.Callable, metrics: object,
                           dev_iterator: base.DataType = None, dev_metrics: object = None, clip: float = None,
                           patience: int = 10, low_is_good: bool = True, shuffle: bool = True, gpu: bool = True,
                           **kwargs) -> base.Union[list, int, dict, dict]:
    """
    Train a single task pytorch model.

    :model (base.ModelType): Untrained model to be trained.
    :save_path (str): Path to save models to.
    :epochs (int): The number of epochs to run.
    :iterator (base.DataType): Batched training set.
    :loss_func (base.Callable): Loss function to use.
    :optimizer (bas.Callable): Optimizer function.
    :metrics (object): Initialized Metrics object.
    :dev_iterator (base.DataType, optional): Batched dev set.
    :dev_metrics (object): Initialized Metrics object.
    :clip (float, default = None): Clip gradients to prevent exploding gradient problem.
    :patience (int, default = 10): Number of iterations to keep going before early stopping.
    :low_is_good (bool, default = False): Lower scores indicate better performance.
    :shuffle (bool, default = True): Shuffle the dataset.
    :gpu (bool, default = True): Run on GPU
    """
    with trange(epochs, desc = "Training epochs", leave = False) as loop:
        preds, labels, loss = [], [], []

        dev_losses = []

        if patience > 0:
            early_stopping = EarlyStopping(save_path, patience, low_is_good = low_is_good)

        for ep in loop:
            model.train()
            optimizer.zero_grad()  # Zero out gradients

            if shuffle:
                iterator.shuffle()

            epoch_preds, epoch_labels, epoch_loss = _singletask_epoch(model, optimizer, loss_func, iterator, clip, gpu)
            epoch_loss = np.mean(epoch_loss)

            preds.extend(epoch_preds)
            labels.extend(epoch_labels)
            loss.append(epoch_loss)
            metrics.compute(epoch_labels, epoch_preds)

            try:
                dev_loss, _, _, _ = eval_torch_model(model, dev_iterator, loss_func, dev_metrics, gpu, store = False,
                                                     **kwargs)
                dev_losses.append(dev_loss)
                dev_score = dev_metrics[dev_metrics.display_metric][-1]

                if early_stopping is not None and early_stopping(model, dev_loss):
                    early_stopping.set_best_state(model)
                    break

                loop.set_postfix(loss = f"{epoch_loss:.4f}", dev_loss = f"{dev_loss:.4f}",
                                 **metrics.display(), dev_score = dev_score)
            except Exception:
                # TODO Add logging of error
                loop.set_postfix(loss = f"{epoch_loss:.4f}", **metrics.display())

    return loss, dev_losses, metrics.scores, dev_metrics.scores


def _mtl_epoch(model: base.ModelType, loss_func: base.Callable, loss_weights: base.DataType, opt: base.Callable,
               metrics: object, batchers: base.List[base.Batch], batch_count: int, dataset_weights: base.List[float],
               clip: float = None, **kwargs):
    """
    Train one epoch of an MTL training loop.

    :model (base.ModelType): Model in the process of being trained.
    :loss_func (base.Callable): The loss function being used.
    :loss_weights (base.DataType): Determines relative task importance When using multiple input/output functions.
    :opt (base.Callable): The optimizer function used.
    :metrics (object): Initialized Metrics object.
    :batchers (base.List[base.Batch]): A list of batched objects.
    :batch_count (int): The number of batches to go through in each epoch.
    :dataset_weights (base.List[float]): The probability with which each dataset is chosen to be trained on.
    :clip (float, default = None): Use gradient clipping.
    """
    epoch_loss = []

    with tqdm(range(batch_count, desc = 'Batch', leave = False)) as b:

        # Select task and get batch
        task_id = np.random.choice(range(len(batchers)), p = dataset_weights)
        X, y = next(iter(batchers[task_id]))

        # Do model training
        model.train()
        opt.zero_grad()

        preds = model(X, task_id, **kwargs)
        loss = loss_func(preds, y) * loss_weights[task_id]
        loss.backwards()

        if clip is not None:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)  # Prevent exploding gradients

        opt.step()

        metrics.compute()
        epoch_loss.append(loss.data.item().cpu())

        b.set_postfix(batch_loss = epoch_loss[-1] / len(y), **metrics.display(), task = task_id)

    return epoch_loss


def train_mtl_model(model: base.ModelType, training_datasets: base.List[base.DataType], save_path: str,
                    opt: base.Callable, loss_func: base.Callable, metrics: object, batch_size: int = 64,
                    epochs: int = 2, clip: float = None, patience: int = 10, dev: base.DataType = None,
                    dev_task_id: int = 0, batches_per_epoch: int = None, shuffle_data: bool = True,
                    dataset_weights: base.DataType = None, loss_weights: base.DataType = None, **kwargs) -> None:
    """
    Train a multi-task learning model.

    :model (base.ModelType): Untrained model.
    :training_datasets (base.List[base.DataType]): List of tuples containing dense matrices.
    :save_path (str): Path to save trained model to.
    :opt (base.Callable): Pytorch optimizer to train model.
    :loss_func (base.Callable): Loss function.
    :metrics (object): Initialized metrics object.
    :batch_size (int): Training batch size.
    :epochs (int): Maximum number of epochs (if no early stopping).
    :clip (float, default = None): Use gradient clipping.
    :patience (int, default = 10): Number of epochs to observe non-improving dev performance before early stopping.
    :dev (base.DataType): Dev dataset object.
    :dev_task_id (int, default = 0): Task ID for task to use for early stopping, in case of multitask learning.
    :batches_per_epoch (int, default = None): Set number of batches per epoch. If None, an epoch consists of all
                                              training examples.
    :shuffle_data: Whether to shuffle data at training.
    :dataset_weights (base.DataType, default = None): Probability for each dataset to be chosen (must sum to 1.0).
    :loss_weights (base.DataType): Determines relative task importance When using multiple input/output functions.
    """
    if loss_weights is None:
        loss_weights = np.ones(len(training_datasets))

    if dataset_weights is None:
        dataset_weights = loss_weights / len(training_datasets)

    if batches_per_epoch is None:
        batches_per_epoch = sum([len(dataset) * batch_size for dataset in training_datasets]) // batch_size

    if patience > 0:
        early_stopping = EarlyStopping(save_path, patience, low_is_good = False)

    batchers = []

    for train_data in training_datasets:
        batches = process_and_batch(train_data, train_data.data, batch_size, 'label')

        if shuffle_data:
            batches.shuffle()

        batchers.append(batches)

    with trange(epochs, desc = "Training model") as t:
        dev_scores = []
        dev_losses = []
        train_loss = []

        for epoch in t:
            epoch_loss = _mtl_epoch(model, loss_func, loss_weights, opt, batchers, batches_per_epoch,
                                    dataset_weights, clip)
            epoch_loss = epoch_loss / len(epoch_loss)
            train_loss.append(epoch_loss)

            try:
                dev_batches = process_and_batch(dev, dev.dev, len(dev.dev))
                dev_loss, _, dev_score, _ = eval_torch_model(model, dev_batches, loss_func,
                                                              metrics, mtl = True,
                                                              task_id = dev_task_id)
                dev_loss = dev_loss / len(dev_loss)
                dev_losses.append(dev_loss)
                dev_scores.append(dev_score)

                t.set_postfix(epoch_loss = epoch_loss, dev_loss = dev_loss)

                if early_stopping is not None and early_stopping(model, dev_scores.early_stopping()):
                    early_stopping.set_best_state(model)
                    break

            except Exception:
                t.set_postfix(epoch_loss = epoch_loss)
            finally:
                t.refresh()
    return train_loss, dev_losses, _, dev_scores


def train_sklearn_model(arg1):
    """
    TODO: Docstring for train_sklearn_model.

    :arg1: TODO
    :returns: TODO

    """
    train_scores, dev_scores = None, None
    raise NotImplementedError
    return None, None, train_scores, dev_scores


def evaluate_sklearn_model(arg1):
    """
    TODO: Docstring for evaluate_sklearn_model.

    :arg1: TODO
    :returns: TODO

    """
    train_scores, dev_scores = None, None
    raise NotImplementedError
    return None, None, train_scores, dev_scores
