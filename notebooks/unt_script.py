# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
import csv
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch

from torch.optim import Adam
from mlearn.base import Field
from mlearn.data.fileio import *
from mlearn import base
from mlearn.data import clean
from mlearn.data.clean import Cleaner, Preprocessors
from mlearn.data import loaders
from mlearn.utils.metrics import Metrics
from mlearn.modeling.multitask import EmbeddingLSTMClassifier
from mlearn.data.dataset import GeneralDataset
from mlearn.utils.early_stopping import EarlyStopping
from mlearn.utils.pipeline import process_and_batch 
from mlearn.utils.train import train_mtl_model

# %% [markdown]
# # Load data

# %%
cl = Cleaner(processes = ['lower', 'url', 'hashtag'])
pr = Preprocessors(liwc_dir = '/Users/pranavsood/Documents/GitHub/mlearn/Multitask-Abuse/data/')
m = Metrics(['accuracy'], 'accuracy')


# %%
## Slow version
davidson = loaders.oraby_sarcasm(cleaners = cl.tokenize , data_path = '/Users/pranavsood/Documents/GitHub/mlearn/Multitask-Abuse/data/json', length = 200, label_processor = None)


# %%
## Slow version
hoover = loaders.oraby_sarcasm(cleaners = cl.tokenize , data_path = '/Users/pranavsood/Documents/GitHub/mlearn/Multitask-Abuse/data/json', length = 200,
                        preprocessor = pr.word_token, label_processor = lambda x: x.split()[0])

# %% [markdown]
# # Process data

# %%
# Davidson
davidson.build_token_vocab(davidson.data)
davidson.build_label_vocab(davidson.data)


# %%
hoover.build_token_vocab(hoover.data)
hoover.build_label_vocab(hoover.data)


# %%
print(hoover.ltoi)
print(davidson.ltoi)
print(hoover.vocab_size())
print(davidson.vocab_size())
print(hoover.ltoi)
print(davidson.ltoi)
print(hoover.data[0].__dict__)
print(davidson.data[0].__dict__)


# %%
from mlearn.utils.pipeline import process_and_batch

processed_dav_tr = process_and_batch(davidson, davidson.data, 32, onehot = False)
processed_hoo_tr = process_and_batch(hoover, hoover.data, 32, onehot = False)
processed_hoo_de = process_and_batch(hoover, hoover.dev, 32, onehot = False)


# %%
model = EmbeddingLSTMClassifier(input_dims = [int(hoover.vocab_size()), int(davidson.vocab_size())], shared_dim = 150,
                          hidden_dims = [128, 128], output_dims = [hoover.label_count(), davidson.label_count()],
                          no_layers = 1, dropout = 0.2, embedding_dims = 128)


# %%
optimizer = Adam(model.parameters(), lr=0.1)
loss = nn.NLLLoss()
model.name


# %%
def _mtl_epoch(model: base.ModelType, loss_f: base.Callable, loss_weights: base.DataType, optimizer: base.Callable,
               metrics: object, batchers: base.List[base.Batch], batch_count: int, dataset_weights: base.List[float],
               taskid2name: dict, epoch_no: int, clip: float = None, gpu: bool = True, **kwargs) -> None:
    """
    Train one epoch of an MTL training loop.

    :model (base.ModelType): Model in the process of being trained.
    :loss_f (base.Callable): The loss function being used.
    :loss_weights (base.DataType): Determines relative task importance When using multiple input/output functions.
    :optimizer (base.Callable): The optimizer function used.
    :metrics (object): Initialized Metrics object.
    :batchers (base.List[base.Batch]): A list of batched objects.
    :batch_count (int): The number of batchers to go through in each epoch.
    :dataset_weights (base.List[float]): The probability with which each dataset is chosen to be trained on.
    :taskid2name (dict): Dictionary mapping task ID to dataset name.
    :epoch_no (int): The iteration of the epoch.
    :clip (float, default = None): Use gradient clipping.
    """
    with tqdm(range(batch_count), desc = 'Batch', leave = False) as loop:
        label_count = 0
        epoch_loss = 0

        for i, b in enumerate(loop):
            # Select task and get batch
            task_id = np.random.choice(range(len(batchers)), p = dataset_weights)
            X, y = next(iter(batchers[task_id]))

            if gpu:
                X = X.cuda()
                y = y.cuda()

            # Do model training
            model.train()
            optimizer.zero_grad()

            scores = model(X, task_id, **kwargs)
            loss = loss_f(scores, y) * loss_weights[task_id]
            loss.backward()

            if clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)  # Prevent exploding gradients

            optimizer.step()

            metrics.compute(torch.argmax(scores, dim = 1).tolist(), y.tolist())
            label_count += len(y.cpu().tolist())
            epoch_loss += loss.data.item()
            metrics.loss = loss.data.item() / len(y)

            # Write batch info
            task_name = taskid2name[task_id]
            #mtl_batch_writer(model = model, batch = i, metrics = metrics, task_name = task_name, epoch = epoch_no,
            #                 **kwargs)

            loop.set_postfix(batch_loss = f"{metrics.get_last('loss'):.4f}",
                             epoch_loss = f"{epoch_loss / label_count:.4f}",
                             task_score = f"{metrics.last_display():.4f}",
                             task = task_id)


# %%



# %%
from collections import defaultdict
from tqdm import trange

def train_model(model: base.ModelType, batchers: base.List[base.DataType], optimizer: base.Callable,
                    loss: base.Callable, metrics: object, batch_size: int = 64, epochs: int = 2, clip: float = None,
                    earlystop: int = None, save_path: str = None, dev: base.DataType = None, dev_metrics: object = None,
                    dev_task_id: int = 0, batches_per_epoch: int = None, low: bool = True,
                    shuffle: bool = True, dataset_weights: base.DataType = None, loss_weights: base.DataType = None,
                    gpu: bool = True, hyperopt = None, **kwargs) -> None:
    """
    Train a multi-task learning model.

    :model (base.ModelType): Untrained model.
    :batchers (base.List[base.DataType]): Batched training data.
    :save_path (str): Path to save trained model to.
    :optimizer (base.Callable): Pytorch optimizer to train model.
    :loss (base.Callable): Loss function.
    :metrics (object): Initialized metrics object.
    :batch_size (int): Training batch size.
    :epochs (int): Maximum number of epochs (if no early stopping).
    :clip (float, default = None): Use gradient clipping.
    :earlystop (int, default = None): Number of epochs to observe non-improving dev performance before early stopping.
    :dev (base.DataType): Batched dev object.
    :dev_metrics (object): Initialized dev_metrics object.
    :dev_task_id (int, default = 0): Task ID for task to use for early stopping, in case of multitask learning.
    :batches_per_epoch (int, default = None): Set number of batchers per epoch. If None, an epoch consists of all
                                              training examples.
    :low (bool, default = True): If lower value is to be interpreted as better by EarlyStopping.
    :shuffle: Whether to shuffle data at training.
    :dataset_weights (base.DataType, default = None): Probability for each dataset to be chosen (must sum to 1.0).
    :loss_weights (base.DataType, default = None): Weight the loss by multiplication.
    :gpu (bool, default = True): Set tot rue if model runs on GPU.
    :hyperopt (default = None): Trial object for hyper parameter search.
    """
    with trange(epochs, desc = "Training model", leave = False) as loop:
        taskid2name = {i: batchers[i].data.name for i in range(len(batchers))}
        scores = defaultdict(list)

        if gpu:
            model = model.cuda()

        if loss_weights is None:
            loss_weights = np.ones(len(batchers))

        if dataset_weights is None:
            dataset_weights = np.ones(len(batchers)) / len(batchers)

        if batches_per_epoch is None:
            batches_per_epoch = sum([len(dataset) * batch_size for dataset in batchers]) // batch_size

        if earlystop is not None:
            earlystop = EarlyStopping(save_path, model, earlystop, low_is_good = low)

        for i, epoch in enumerate(loop):
            if shuffle:
                for batch in batchers:
                    batch.shuffle()

            _mtl_epoch(model, loss, loss_weights, optimizer, metrics, batchers, batches_per_epoch, dataset_weights,
                       taskid2name, i, clip, gpu = gpu, **kwargs)

            for score in metrics.scores:  # Compute average value of the scores computed in each epoch.
                if score == 'loss':
                    scores[score].append(sum(metrics.scores[score]))
                else:
                    scores[score].append(np.mean(metrics.scores[score]))

            try:
                eval_torch_model(model, dev, loss, dev_metrics, mtl = dev_task_id, store = False, gpu = gpu, **kwargs)

                loop.set_postfix(loss = f"{metrics.get_last('loss'):.4f}",
                                 dev_loss = f"{dev_metrics.get_last('loss'):.4f}",
                                 dev_score = f"{dev_metrics.last_display():.4f}")

                if hyperopt:
                    hyperopt.report(dev_metrics.last_display(), epoch)

                if earlystop is not None and earlystop(model, dev_metrics.early_stopping()):
                    model = earlystop.best_state
                    break
            except Exception:
                loop.set_postfix(epoch_loss = metrics.get_last('loss'))
            finally:
                loop.refresh()
        metrics.scores = scores


# %%
def _mtl_epoch(model: base.ModelType, loss_f: base.Callable, loss_weights: base.DataType, optimizer: base.Callable,
               metrics: object, batchers: base.List[base.Batch], batch_count: int, dataset_weights: base.List[float],
               taskid2name: dict, epoch_no: int, clip: float = None, gpu: bool = True, **kwargs) -> None:
    """
    Train one epoch of an MTL training loop.

    :model (base.ModelType): Model in the process of being trained.
    :loss_f (base.Callable): The loss function being used.
    :loss_weights (base.DataType): Determines relative task importance When using multiple input/output functions.
    :optimizer (base.Callable): The optimizer function used.
    :metrics (object): Initialized Metrics object.
    :batchers (base.List[base.Batch]): A list of batched objects.
    :batch_count (int): The number of batchers to go through in each epoch.
    :dataset_weights (base.List[float]): The probability with which each dataset is chosen to be trained on.
    :taskid2name (dict): Dictionary mapping task ID to dataset name.
    :epoch_no (int): The iteration of the epoch.
    :clip (float, default = None): Use gradient clipping.
    """
    with tqdm(range(batch_count), desc = 'Batch', leave = False) as loop:
        label_count = 0
        epoch_loss = 0

        for i, b in enumerate(loop):
            # Select task and get batch
            task_id = np.random.choice(range(len(batchers)), p = dataset_weights)
            X, y = next(iter(batchers[task_id]))

            if gpu:
                X = X.cuda()
                y = y.cuda()

            # Do model training
            model.train()
            optimizer.zero_grad()

            scores = model(X, task_id, **kwargs)
            loss = loss_f(scores, y) * loss_weights[task_id]
            loss.backward()

            if clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)  # Prevent exploding gradients

            optimizer.step()

            metrics.compute(torch.argmax(scores, dim = 1).tolist(), y.tolist())
            label_count += len(y.cpu().tolist())
            epoch_loss += loss.data.item()
            metrics.loss = loss.data.item() / len(y)

            # Write batch info
            task_name = taskid2name[task_id]
            #mtl_batch_writer(model = model, batch = i, metrics = metrics, task_name = task_name, epoch = epoch_no,
            #                 **kwargs)

            loop.set_postfix(batch_loss = f"{metrics.get_last('loss'):.4f}",
                             epoch_loss = f"{epoch_loss / label_count:.4f}",
                             task_score = f"{metrics.last_display():.4f}",
                             task = task_id)


# %%
train_model(model, [processed_hoo_tr, processed_dav_tr], optimizer, loss, dev_data = processed_hoo_de, metrics = m, gpu = False) # 2 min 10s with my code , 


# %%



