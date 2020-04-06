import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from mlearn import base
from collections import defaultdict
import mlearn.data_processing.data as data
from mlearn.modeling.metrics import compute
from mlearn.modeling.evaluate import eval_torch_model
from mlearn.modeling.early_stopping import EarlyStopping
from mlearn.data_processing.batching import Batch, BatchExtractor


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


def write_predictions(output_info: pd.DataFrame, main_dataset: data.GeneralDataset, preds: list, truths: list,
                      model_info: list, model_header: list, data_name: str, main_name: str):
    """TODO: Docstring for write_predictions.
    :output_info (pd.DataFrame): Dataframe containing information to be written including each doc.
    :main_dataset (data.GeneralDataset): dataset for main task.
    :preds (list): Predictions
    :model_info (list): Model information
    :model_header (list): Header with field information of the model.
    :data_name (str): Dataset evaluated on.
    :main_name (str): Main task dataset.
    :returns: TODO
    """

    output_info['predictions'] = [main_dataset.label_name_lookup(p) for p in preds]
    output_info['true'] = [main_dataset.label_name_lookup(t) for t in truths]

    for head, info in zip(model_header, model_info):
        output_info[head] = info

    # TODO Figure out a way to get access to the original document after prediction
    # TODO Write all predictions out to a file.
    # TODO File header: Dataset, Model info, Train (yes/no), Predicted label, True label, Document
    pass


def write_results(writer: base.Callable, train_scores: dict, train_loss: list, dev_scores: dict, dev_loss: list,
                  epochs: int, model_info: list, metrics: list, exp_len: int, data_name: str, main_name: str,
                  **kwargs) -> None:
    """Write results to file.
    :writer (base.Callable): Path to file.
    :train_scores (dict): Train scores.
    :train_loss (list): Train losses.
    :dev_scores (dict): Dev scores.
    :dev_loss (list): Dev losses.
    :epochs (int): Epochs.
    :model_info (list): Model info.
    :metrics (list): Model info.
    :exp_len (int): Expected length of each line.
    :data_name (str): Name of the dataset that's being run on.
    :main_name (str): Name of the dataset the model is trained/being trained on.
    """
    for i in range(epochs):
        try:
            out = [data_name, main_name] + [i] + model_info  # Base info
            out += [train_scores[m][i] for m in metrics] + [train_loss[i]]  # Train info
            if dev_scores:
                out += [dev_scores[m][i] for m in metrics] + [dev_loss[i]]  # Dev info
        except IndexError as e:
            __import__('pdb').set_trace()

        row_len = len(out)
        if row_len < exp_len:
            out += [''] * (row_len - exp_len)
        elif row_len > exp_len:
            __import__('pdb').set_trace()

        writer.writerow(out)


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

        # update steps
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
                     clip: base.Union[int, float] = None, **kwargs):
    """Train one epoch of an MTL training loop.
    :model (base.ModelType): Model in the process of being trained.
    :loss_func (base.Callable): The loss function being used.
    :loss_weights (base.DataType): Determines relative task importance When using multiple input/output functions.
    :opt (base.Callable): The optimizer function used.
    :batchers (base.List[base.Batch]): A list of batched objects.
    :batch_count (int): The number of batches to go through in each epoch.
    :dataset_weights (base.List[float]): The probability with which each dataset is chosen to be trained on.
    :clip (base.Union[int, float], default = None): Use gradient clipping.
    """
    epoch_loss = []

    for b in tqdm(range(batch_count), desc = "Iterating over batches"):
        task_id = np.random.choice(range(len(batchers)), p = dataset_weights)  # set probability for each task
        batcher = batchers[task_id]
        X, y = next(iter(batcher))

        # Do model training
        model.train()
        opt.zero_grad()

        preds = model(X, task_id, **kwargs)
        loss = loss_func(preds, y) * loss_weights[task_id]
        loss.backwards()

        if clip is not None:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)  # Prevent exploding gradients

        opt.step()

        epoch_loss.append(loss.data.item().cpu())


def train_mtl_model(model, training_datasets, save_path, optimizer, metrics: base.Dict[str, base.Callable],
                    dev_metric: str, batch_size = 64, epochs = 2, clip = None, dev = None, dev_task_id = 0,
                    dataset_weights = None, patience = 10, batches_per_epoch = None, shuffle_data = True,
                    loss_weights = None, loss_func = None):
    """Trains a multi-task learning model.
    :model: Untrained model.
    :training_datasets: List of tuples containing dense matrices.
    :save_path: Path to save trained model to.
    :optimizer: Pytorch optimizer to train model.
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

    for epoch in tqdm(range(epochs), desc = "Training model"):
        epoch_loss = _train_mtl_epoch(model, loss_func, loss_weights, optimizer, batchers, batches_per_epoch,
                                      dataset_weights, clip)

        print("Epoch train loss:", np.array(epoch_loss).mean())

        if dev is not None:
            dev_batches = process_and_batch(dev, dev.dev, len(dev.dev))
            dev_loss, _, dev_scores, _ = eval_torch_model(model, dev_batches, loss_func,
                                                          metrics, mtl = True,
                                                          task_id = dev_task_id)

            if early_stopping is not None and early_stopping(model, dev_scores[dev_metric]):
                early_stopping.set_best_state(model)
                break


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
