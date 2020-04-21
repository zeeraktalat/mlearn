import torch
import numpy as np
from tqdm import tqdm, trange
from mlearn import base
from collections import defaultdict
from mlearn.modeling.evaluate import eval_torch_model
from mlearn.modeling.early_stopping import EarlyStopping
from mlearn.data_processing.batching import Batch, BatchExtractor
from mlearn.data_processing.fileio import write_predictions, write_results


def process_and_batch(dataset, data, batch_size: int, onehot: bool = True):
    """
    Process a dataset and data.

    :dataset: A dataset object.
    :data: Data to be processed.
    :batch_size (int): Size of batches to create.
    :returns: Processed data.
    """
    # Process labels and encode data.
    dataset.process_labels(data)

    # Batch data
    batch = Batch(batch_size, data)
    batch.create_batches()
    batches = BatchExtractor('label', batch, dataset, onehot)
    return batches


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
        write_predictions(kwargs['iterator'], model_info = model_info, **kwargs)


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

    with tqdm(iterator, desc = "Batch") as loop:

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

            loop.set_postfix(batch_loss = epoch_loss[-1])

    return predictions, labels, epoch_loss


def train_singletask_model(model: base.ModelType, save_path: str, epochs: int, iterator: base.DataType,
                           loss_func: base.Callable, optimizer: base.Callable, metrics: object,
                           dev_iterator: base.DataType = None, clip: float = None, patience: int = 10,
                           shuffle: bool = True, gpu: bool = True, **kwargs) -> base.Union[list, int, dict, dict]:
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
    :clip (float, default = None):
    :shuffle (bool, default = True): Shuffle the dataset.
    :gpu (bool, default = True): Run on GPU
    """
    with trange(epochs) as loop:
        preds, labels, loss = [], [], []
        train_scores = defaultdict(list)

        dev_losses = []
        dev_scores = defaultdict(list)

        if patience > 0:
            early_stopping = EarlyStopping(save_path, patience, low_is_good = False)

        for ep in loop:
            model.train()
            optimizer.zero_grad()  # Zero out gradients

            if shuffle:
                iterator.shuffle()

            epoch_preds, epoch_labels, epoch_loss = _singletask_epoch(model, optimizer, loss_func, iterator, clip, gpu)
            metrics.compute(epoch_labels, epoch_preds)
            epoch_display = metrics.display_metric()

            preds.extend(epoch_preds)
            labels.extend(epoch_labels)
            loss.append(sum(epoch_loss))

            try:
                dev_loss, _, dev_scores, _ = eval_torch_model(model, dev_iterator, loss_func, metrics, **kwargs)
                dev_losses.append(dev_loss)
                dev_score = dev_scores[metrics.display]

                if early_stopping is not None and early_stopping(model, dev_scores[metrics.early_stopping()]):
                    early_stopping.set_best_state(model)
                    break

                ep.set_postfix(loss = epoch_loss, dev_loss = dev_loss, **epoch_display, dev_score = dev_score)
            except Exception:
                # Add logging of error
                ep.set_postfix(loss = epoch_loss, **epoch_display)
            finally:
                loop.refresh()

        train_scores = metrics.compute(preds, labels)

    return loss, dev_losses, train_scores, dev_scores


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

    with tqdm(range(batch_count, desc = 'Batch')) as b:

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

        b.set_postfix(batch_loss = epoch_loss[-1], **metrics.display_metric(), task = task_id)

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
        for epoch in t:
            epoch_loss = _mtl_epoch(model, loss_func, loss_weights, opt, batchers, batches_per_epoch,
                                    dataset_weights, clip)

            try:
                dev_batches = process_and_batch(dev, dev.dev, len(dev.dev))
                dev_loss, _, dev_scores, _ = eval_torch_model(model, dev_batches, loss_func,
                                                              metrics, mtl = True,
                                                              task_id = dev_task_id)

                t.set_postfix(epoch_loss = epoch_loss, dev_loss = dev_loss)

                if early_stopping is not None and early_stopping(model, dev_scores.early_stopping()):
                    early_stopping.set_best_state(model)
                    break

            except Exception:
                t.set_postfix(epoch_loss = epoch_loss)
            finally:
                t.refresh()


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
