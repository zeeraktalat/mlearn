import torch
import numpy as np
from tqdm import tqdm, trange
from mlearn import base
from collections import defaultdict
from mlearn.modeling.metrics import compute
from mlearn.modeling.evaluate import eval_torch_model
from mlearn.modeling.early_stopping import EarlyStopping
from mlearn.data_processing.batching import Batch, BatchExtractor
from mlearn.data_processing.fileio import write_predictions, write_results


def process_and_batch(dataset, data, batch_size: int, onehot: bool = True):
    """Process a dataset and data.

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


def run_model(library: str, train: bool, writer: base.Callable, model_info: list, head_len: int, **kwargs):
    """Train or evaluate model.

    :library (str): Library of the model.
    :train (bool): Whether it's a train or test run.
    :writer (csv.writer): File to output model performance to.
    :model_info (list): Information about the model to be added to each line of the output.
    :head_len (int): The length of the header.
    """
    if train:
        func = train_pytorch_model if library == 'pytorch' else train_sklearn_model
    else:
        func = eval_torch_model if library == 'pytorch' else evaluate_sklearn_model

    train_loss, dev_loss, train_scores, dev_scores = func(**kwargs)
    write_results(writer, train_scores, train_loss, dev_scores, dev_loss, model_info = model_info, exp_len = head_len,
                  **kwargs)

    if not train:
        write_predictions(kwargs['iterator'], model_info = model_info, **kwargs)


def train_epoch(model: base.ModelType, optimizer: base.Callable, loss_func: base.Callable, batches: base.DataType,
                gpu: bool = True, **kwargs):
    """Basic training procedure for pytorch models.

    :model (base.ModelType): Untrained model to be trained.
    :optimizer (bas.Callable): Optimizer function.
    :loss_func (base.Callable): Loss function to use.
    :batches (base.DataType): Batched training set.
    :gpu (bool, default = True): Run on GPU
    :returns: TODO
    """
    predictions, labels = [], []
    epoch_loss = []
    for X, y in tqdm(batches, desc = "iterating over batches", leave = False):

        if gpu:  # make sure it's gpu runnable
            X = X.cuda()
            y = y.cuda()

        scores = model(X, **kwargs)

        loss = loss_func(scores, y)
        epoch_loss.append(float(loss.data.item()))

        # Update steps
        loss.backward()
        optimizer.step()

        predictions.extend(torch.argmax(scores, 1).cpu().tolist())
        labels.extend(y.cpu().tolist())

    return predictions, labels, loss


def train_pytorch_model(model: base.ModelType, epochs: int, batches: base.DataType, loss_func: base.Callable,
                        optimizer: base.Callable, metrics: base.Dict[str, base.Callable],
                        dev_batches: base.DataType = None, gpu: bool = True, shuffle: bool = True,
                        display_metric: str = 'accuracy', **kwargs) -> base.Union[list, int, dict, dict]:
    """Train a machine learning model.

    :model (base.ModelType): Untrained model to be trained.
    :epochs (int): The number of epochs to run.
    :batches (base.DataType): Batched training set.
    :loss_func (base.Callable): Loss function to use.
    :optimizer (bas.Callable): Optimizer function.
    :metrics (base.Dict[str, base.Callable])): Metrics to use.
    :dev_batches (base.DataType, optional): Batched dev set.
    :gpu (bool, default = True): Run on GPU
    :display_metric (str): Metric to be diplayed in TQDM iterator
    """
    model.train()

    train_loss = []
    train_scores = defaultdict(list)

    dev_losses = []
    dev_scores = defaultdict(list)

    for epoch in tqdm(range(epochs), desc = "Training model"):  # TODO Get TQDM to show the scores for each epoch

        optimizer.zero_grad()  # Zero out gradients
        epoch_loss = []

        if shuffle:
            batches.shuffle()

        epoch_preds, epoch_labels, epoch_loss = train_epoch(model, optimizer, loss_func, batches, gpu)

        epoch_scores = compute(metrics, epoch_labels, epoch_preds)
        train_loss.append(sum(epoch_loss))
        # epoch_performance = epoch_scores[display_metric]  TODO

        for metric in metrics:
            train_scores[metrics].append(epoch_scores[metric])

        if dev_batches is not None:
            dev_loss, _, dev_score, _ = eval_torch_model(model, dev_batches, loss_func, metrics, **kwargs)
            dev_losses.append(dev_loss)

            for score in dev_score:
                dev_scores[score].append(dev_score[score])
            # dev_performance = dev_performance[display_metric]  TODO

    return train_loss, dev_losses, train_scores, dev_scores


def _train_mtl_epoch(model: base.ModelType, loss_func: base.Callable, loss_weights: base.DataType, opt: base.Callable,
                     batchers: base.List[base.Batch], batch_count: int, dataset_weights: base.List[float],
                     clip: float = None, **kwargs):
    """Train one epoch of an MTL training loop.

    :model (base.ModelType): Model in the process of being trained.
    :loss_func (base.Callable): The loss function being used.
    :loss_weights (base.DataType): Determines relative task importance When using multiple input/output functions.
    :opt (base.Callable): The optimizer function used.
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


def train_mtl_model(model: base.ModelType, training_datasets: list[base.DataType], save_path: str, opt: base.Callable,
                    metrics: base.Dict[str, base.Callable], dev_metric: str, batch_size = 64, epochs = 2,
                    clip: float = None, dev = None, dev_task_id = 0, dataset_weights = None, patience = 10,
                    batches_per_epoch = None, shuffle_data = True, loss_weights = None, loss_func = None):
    """Trains a multi-task learning model.

    :model: Untrained model.
    :training_datasets: List of tuples containing dense matrices.
    :save_path: Path to save trained model to.
    :opt: Pytorch optimizer to train model.
    :batch_size: Training batch size.
    :patience: Number of epochs to observe non-improving dev performance before early stopping.
    :epochs: Maximum number of epochs (if no early stopping).
    :dev: Dev dataset object.
    :dev_task_id: Task ID for task to use for early stopping, in case of multitask learning.
    :clip: Use gradient clipping.
    :batches_per_epoch: Set fixed number of batches per epoch. If None, an epoch consists of all training examples.
    :shuffle_data: Whether to shuffle data at training.
    :loss_weights (base.DataType): Determines relative task importance When using multiple input/output functions.
    """
    if loss_weights is None:
        loss_weights = np.ones(len(training_datasets))

    if dataset_weights is None:
        dataset_weights = loss_weights / len(training_datasets)

    if batches_per_epoch is None:
        batches_per_epoch = sum([len(dataset) * batch_size for dataset
                                 in training_datasets]) // batch_size
    if patience > 0:
        early_stopping = EarlyStopping(save_path, patience,
                                       low_is_good=False)

    batchers = []

    for train_data in training_datasets:
        batches = process_and_batch(train_data, train_data.data, batch_size, 'label')

        if shuffle_data:
            batches.shuffle()

        batchers.append(batches)

    with trange(epochs, desc = "Training model") as t:
        for epoch in t:
            epoch_loss = _train_mtl_epoch(model, loss_func, loss_weights, opt, batchers, batches_per_epoch,
                                          dataset_weights, clip)

            try:
                dev_batches = process_and_batch(dev, dev.dev, len(dev.dev))
                dev_loss, _, dev_scores, _ = eval_torch_model(model, dev_batches, loss_func,
                                                              metrics, mtl = True,
                                                              task_id = dev_task_id)

                t.set_postfix(epoch_loss = epoch_loss, dev_loss = dev_loss)
                t.refresh()

                if early_stopping is not None and early_stopping(model, dev_scores[dev_metric]):
                    early_stopping.set_best_state(model)
                    break
            except Exception as e:
                t.set_postfix(epoch_loss = epoch_loss)
            finally:
                t.refresh()


def train_sklearn_model(arg1):
    """TODO: Docstring for train_sklearn_model.

    :arg1: TODO
    :returns: TODO

    """
    train_scores, dev_scores = None, None
    raise NotImplementedError
    return None, None, train_scores, dev_scores


def evaluate_sklearn_model(arg1):
    """TODO: Docstring for evaluate_sklearn_model.

    :arg1: TODO
    :returns: TODO

    """
    train_scores, dev_scores = None, None
    raise NotImplementedError
    return None, None, train_scores, dev_scores
