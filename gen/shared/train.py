import torch
import numpy as np
import pandas as pd
from . import base
from tqdm import tqdm
from collections import defaultdict
import gen.shared.data as data
from gen.shared.batching import Batch, BatchExtractor


def process_and_batch(dataset, data, batch_size):
    """Process a dataset and data.
    :dataset: A dataset object.
    :data: Data to be processed.
    :returns: Processed data.
    """
    # Process labels and encode data.
    dataset.process_labels(data)
    encoded = dataset.encode(data, onehot = True)

    # Batch data
    batch = Batch(batch_size, encoded)
    batch.create_batches()
    batches = BatchExtractor('encoded', 'label', batch)

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
                  epochs: int, model_info: list, metrics: list, exp_len: int, data_name: str, **kwargs) -> None:
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
    :data_name (str): Dataset object.
    """
    for i in range(epochs):
        try:
            out = [data_name] + [i] + model_info  # Base info
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
        func = evaluate_pytorch_model if library == 'pytorch' else evaluate_sklearn_model

    train_loss, dev_loss, train_scores, dev_scores = func(**kwargs)
    write_results(writer, train_scores, train_loss, dev_scores, dev_loss, model_info = model_info, exp_len = head_len,
                  **kwargs)


def train_pytorch_model(model: base.ModelType, epochs: int, batches: base.DataType, loss_func: base.Callable,
                        optimizer: base.Callable, metrics: base.Dict[str, base.Callable],
                        dev_batches: base.DataType = None,
                        display_metric: str = 'accuracy', **kwargs) -> base.Union[list, int, dict, dict]:
    """Train a machine learning model.
    :model (base.ModelType): Untrained model to be trained.
    :epochs (int): The number of epochs to run.
    :batches (base.DataType): Batched training set.
    :loss_func (base.Callable): Loss function to use.
    :optimizer (bas.Callable): Optimizer function.
    :metrics (base.Dict[str, base.Callable])): Metrics to use.
    :dev_batches (base.DataType, optional): Batched dev set.
    :display_metric (str): Metric to be diplayed in TQDM iterator
    """

    model.train_mode = True

    train_loss = []
    train_scores = defaultdict(list)

    dev_losses = []
    dev_scores = defaultdict(list)

    for epoch in tqdm(range(epochs)):  # TODO Get TQDM to show the scores for each epoch

        model.zero_grad()  # Zero out gradients
        epoch_loss = []
        epoch_scores = defaultdict(list)

        for X, y in batches:
            scores = model(X)

            loss = loss_func(scores, y)
            epoch_loss.append(float(loss.item()))

            # Update steps
            loss.backward()
            optimizer.step()

            scores = torch.argmax(scores, 1)
            for metric, scorer in metrics.items():
                performance = scorer(scores, y)
                epoch_scores[metric].append(performance)

        # epoch_performance = np.mean(epoch_scores[display_metric])  TODO

        train_loss.append(sum(epoch_loss))

        for metric in metrics:
            train_scores[metric].append(np.mean(epoch_scores[metric]))

        if dev_batches:
            dev_loss, _, dev_score, _ = evaluate_pytorch_model(model, dev_batches, loss_func, metrics)
            dev_losses.extend(dev_loss)

            for score in dev_score:
                dev_scores[score].extend(dev_score[score])
            # dev_performance = dev_performance[display_metric]  TODO

    return train_loss, dev_losses, train_scores, dev_scores


def evaluate_pytorch_model(model: base.ModelType, iterator: base.DataType, loss_func: base.Callable,
                           metrics: base.Dict[str, base.Callable], **kwargs) -> base.List[float]:
    """Evaluate a machine learning model.
    :model (base.ModelType): Untrained model to be trained.
    :iterator (base.DataType): Test set to evaluate on.
    :loss_func (base.Callable): Loss function to use.
    :metrics (base.Dict[str, base.Callable])): Metrics to use.
    """
    model.train_mode = False
    loss = []
    eval_scores = defaultdict(list)

    with torch.no_grad():
        for X, y in iterator:
            scores = model(X)

            loss_f = loss_func(scores, y)

            scores = torch.argmax(scores, 1)
            for metric, scorer in metrics.items():
                performance = scorer(scores, y)
                eval_scores[metric].append(performance)

            loss.append(loss_f.item())

    return [np.mean(loss)], None, {m: [np.mean(vals)] for m, vals in eval_scores.items()}, None


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
