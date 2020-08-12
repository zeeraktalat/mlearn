import torch
import numpy as np
from mlearn import base
from tqdm import tqdm, trange
from collections import defaultdict
from mlearn.utils.metrics import Metrics
from mlearn.utils.early_stopping import EarlyStopping
from mlearn.utils.evaluate import eval_torch_model, eval_sklearn_model
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from mlearn.data.fileio import write_predictions, write_results, mtl_batch_writer


def _singletask_epoch(model: base.ModelType, optimizer: base.Callable, loss_f: base.Callable, metrics: Metrics,
                      batchers: base.DataType, clip: float = None, gpu: bool = True, **kwargs):
    """
    Training procedure for single task pytorch models.

    :model (base.ModelType): Untrained model to be trained.
    :optimizer (bas.Callable): Optimizer function.
    :loss_f (base.Callable): Loss function to use.
    :batchers (base.DataType): Batched training set.
    :clip (float, default = None): Add gradient clipping to prevent exploding gradients.
    :gpu (bool, default = True): Run on GPU
    :returns: TODO
    """
    with tqdm(batchers, desc = "Batch", leave = False) as loop:
        predictions, labels = [], []
        epoch_loss = 0

        for X, y in loop:
            if gpu:
                X = X.cuda()
                y = y.cuda()

            scores = model(X, **kwargs)
            loss = loss_f(scores, y)
            loss.backward()

            if clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)

            optimizer.step()

            predictions.extend(torch.argmax(scores, dim = 1).cpu().tolist())
            labels.extend(y.cpu().tolist())

            lo = loss.data.item()
            epoch_loss += lo

            loop.set_postfix(batch_loss = f"{lo / len(y) :.4f}")
        metrics.compute(labels, predictions)
        metrics.loss = epoch_loss / len(labels)


def train_singletask_model(model: base.ModelType, save_path: str, epochs: int, batchers: base.DataType,
                           loss: base.Callable, optimizer: base.Callable, metrics: Metrics,
                           dev: base.DataType = None, dev_metrics: Metrics = None, clip: float = None,
                           early_stopping: int = None, low: bool = False, shuffle: bool = True, gpu: bool = True,
                           hyperopt = None, **kwargs) -> base.Union[list, int, dict, dict]:
    """
    Train a single task pytorch model.

    :model (base.ModelType): Untrained model to be trained.
    :save_path (str): Path to save models to.
    :epochs (int): The number of epochs to run.
    :batchers (base.DataType): Batched training set.
    :loss (base.Callable): Loss function to use.
    :optimizer (bas.Callable): Optimizer function.
    :metrics (object): Initialized Metrics object.
    :dev (base.DataType, optional): Batched dev set.
    :dev_metrics (object): Initialized Metrics object.
    :clip (float, default = None): Clip gradients to prevent exploding gradient problem.
    :early_stopping (int, default = 10): Number of iterations to keep going before early stopping.
    :low (bool, default = False): Lower scores indicate better performance.
    :shuffle (bool, default = True): Shuffle the dataset.
    :gpu (bool, default = True): Run on GPU
    :hyperopt (default = None): Do hyper parameter optimisation.
    """
    with trange(epochs, desc = "Training epochs", leave = False) as loop:
        if gpu:
            model = model.cuda()

        if early_stopping is not None:
            early_stopping = EarlyStopping(save_path, model, early_stopping, low_is_good = low)

        for ep in loop:
            model.train()
            optimizer.zero_grad()  # Zero out gradients

            if shuffle:
                batchers.shuffle()

            _singletask_epoch(model, optimizer, loss, metrics, batchers, clip, gpu)

            try:
                eval_torch_model(model, dev, loss, dev_metrics, gpu, store = False, **kwargs)

                loop.set_postfix(epoch_loss = f"{metrics.get_last('loss'):.4f}",
                                 dev_loss = f"{dev_metrics.get_last('loss'):.4f}",
                                 **metrics.display(),
                                 dev_score = f"{dev_metrics.last_display():.4f}")

                if hyperopt:
                    hyperopt.report(dev_metrics.last_display(), ep)

                if early_stopping is not None and early_stopping(model, dev_metrics.early_stopping()):
                    model = early_stopping.best_state
                    break
            except Exception:
                # Dev is not set.
                loop.set_postfix(epoch_loss = f"{metrics.get_last('loss'):.4f}", **metrics.display())
            finally:
                loop.refresh()


def run_singletask_model(train: bool, writer: base.Callable, pred_writer: base.Callable = None,
                         library: str = 'pytorch', **kwargs) -> None:
    """
    Train or evaluate model.

    :train (bool): Whether it's a train or test run.
    :writer (csv.writer): File to output model performance to.
    :pred_writer (base.Callable): File to output the model predictions to.
    :library (str): Library of the model.
    """
    if train:
        func = train_singletask_model if library == 'pytorch' else select_sklearn_training_regiment
    else:
        func = eval_torch_model if library == 'pytorch' else eval_sklearn_model

    func(**kwargs)
    write_results(writer, **kwargs)

    if not train and pred_writer is not None:
        write_predictions(pred_writer, **kwargs)


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
            mtl_batch_writer(model = model, batch = i, metrics = metrics, task_name = task_name, epoch = epoch_no,
                             **kwargs)

            loop.set_postfix(batch_loss = f"{metrics.get_last('loss'):.4f}",
                             epoch_loss = f"{epoch_loss / label_count:.4f}",
                             task_score = f"{metrics.last_display():.4f}",
                             task = task_id)


def train_mtl_model(model: base.ModelType, batchers: base.List[base.DataType], optimizer: base.Callable,
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


def run_mtl_model(train: bool, writer: base.Callable, pred_writer: base.Callable = None, library: str = 'pytorch',
                  **kwargs) -> None:
    """
    Train or evaluate model.

    :train (bool): Whether it's a train or test run.
    :writer (csv.writer): File to output model performance to.
    :pred_writer (base.Callable): File to output the model predictions to.
    :library (str): Library of the model.
    """
    if train:
        func = train_mtl_model if library == 'pytorch' else select_sklearn_training_regiment
    else:
        func = eval_torch_model if library == 'pytorch' else eval_sklearn_model

    func(**kwargs)

    write_results(writer, **kwargs)

    if not train and pred_writer is not None:
        write_predictions(pred_writer, **kwargs)


def train_sklearn_cv_model(model: base.ModelType, vectorizer: base.VectType, dataset: base.DataType,
                           cross_validate: int = None, stratified: bool = True, metrics: Metrics = None,
                           dev: base.DataType = None, dev_metrics: Metrics = None, **kwargs
                           ) -> base.Tuple[Metrics, Metrics]:
    """
    Train sklearn cv model.

    :model (base.ModelType): An untrained scikit-learn model.
    :vectorizer (base.VectType): An unfitted vectorizer.
    :dataset (GeneralDataset): The dataset object containing the training set.
    :cross_validate (int, default = None): The number of folds for cross-validation.
    :stratified (bool, default = True): Stratify data across the folds.
    :metrics (Metrics, default = None): An initialized metrics object.
    :dev (base.DataType, default = None): The development data.
    :dev_metrcs (Metrics, default = None): An initialized metrics object for the dev data.
    :returns (model, metrics, dev_metrics): Returns trained model object, metrics and dev_metrics objects.
    """
    # Load data
    train = dataset.vectorize(dataset.train, dataset, vectorizer)
    labels = [doc.label for doc in dataset.train]

    if stratified:
        folds = StratifiedKFold(cross_validate)
    else:
        folds = KFold(cross_validate)

    with trange(folds, desc = "Training model") as loop:
        for train_idx, test_idx in folds.split(train, labels):
            trainX, trainY = train[train_idx], labels[train_idx]
            testX, testY = train[test_idx], labels[test_idx]

            model.fit(trainX, trainY)
            eval_sklearn_model(model, testX, metrics, testY)

        try:
            devX = dataset.vectorize(dev, dataset, vectorizer)
            devY = [getattr(doc, getattr(f, 'name')) for f in dataset.label_fields for doc in dev]
            eval_sklearn_model(model, devX, dev_metrics, devY)

            loop.set_postfix(**metrics.display(), **dev_metrics.display())
        except Exception:
            loop.set_postfix(**metrics.display())
        finally:
            loop.refresh()

    return model, metrics, dev_metrics


def train_sklearn_gridsearch_model(model: base.ModelType, vectorizer: base.VectType, dataset: base.DataType,
                                   grid_search: dict, cross_validate: int = None, metrics: Metrics = None,
                                   dev: base.DataType = None, dev_metrics: Metrics = None, scoring: str = 'f1_weighted',
                                   n_jobs: int = -1, **kwargs) -> base.Tuple[base.ModelType, Metrics, Metrics]:
    """
    Train sklearn model using grid-search.

    :model (base.ModelType): An untrained scikit-learn model.
    :vectorizer (base.VectType): An unfitted vectorizer.
    :dataset (base.DataType): The dataset object containing train data.
    :grid_search (dict): The parameter grid to explore.
    :cross_validate (int, default = None): The number of folds for cross-validation.
    :metrics (Metrics, default = None): An initialized metrics object.
    :dev (base.DataType, default = None): The development data.
    :dev_metrcs (Metrics, default = None): An initialized metrics object for the dev data.
    :scoring (str, default = 'f1_weighted'): The scoring metrics used to define best functioning model.
    :n_jobs (int, default = -1): The number of processors to use (-1 == all processors).
    :returns (model, metrics, dev_metrics): Returns grid-search object, metrics and dev_metrics objects.
    """
    train = dataset.vectorize(dataset.train, dataset, vectorizer)
    labels = [doc.label for doc in dataset.train]

    with trange(1, desc = "Training model") as loop:
        model = GridSearchCV(model, grid_search, scoring, n_jobs = n_jobs, cv = cross_validate, refit = True)
        model.fit(train, labels)

        try:
            devX = dataset.vectorize(dev, dataset, vectorizer)
            devY = [getattr(doc, getattr(f, 'name')) for f in dataset.label_fields for doc in dev]
            eval_sklearn_model(model, devX, dev_metrics, devY)

            loop.set_postfix(f1_score = model.best_score_, **dev_metrics.display())
        except Exception:
            loop.set_postfix(f1_score = model.best_score_)
        finally:
            loop.refresh()

    return model, metrics, dev_metrics


def train_sklearn_model(model: base.ModelType, vectorizer: base.VectType, dataset: base.DataType, metrics: Metrics,
                        dev: base.DataType = None, dev_metrics: Metrics = None, **kwargs):
    """
    Train bare sci-kit learn model.

    :model (base.ModelType): An untrained scikit-learn model.
    :vectorizer (base.VectType): An unfitted vectorizer.
    :dataset (base.DataType): The dataset object containing train data.
    :grid_search (dict): The parameter grid to explore.
    :cross_validate (int, default = None): The number of folds for cross-validation.
    :metrics (Metrics, default = None): An initialized metrics object.
    :dev (base.DataType, default = None): The development data.
    :dev_metrcs (Metrics, default = None): An initialized metrics object for the dev data.
    :returns (model, metrics, dev_metrics): Returns trained model object, metrics and dev_metrics objects.
    """
    with trange(1, desc = "Training model") as loop:
        trainX = dataset.vectorize(dataset.train, dataset, vectorizer)
        trainY = [doc.label for doc in dataset.train]

        model.fit(trainX, trainY)

        try:
            devX = dataset.vectorize(dev, dataset, vectorizer)
            devY = [getattr(doc, getattr(f, 'name')) for f in dataset.label_fields for doc in dev]
            eval_sklearn_model(model, devX, dev_metrics, devY)

            loop.set_postfix(**metrics.display(), **dev_metrics.display())
        except Exception:
            loop.set_postfix(**metrics.display())
        finally:
            loop.refresh()

    return model, metrics, dev_metrics


def select_sklearn_training_regiment(model: base.ModelType, cross_validate: int = None, grid_search: dict = None,
                                     **kwargs):
    """
    Select the type of sklearn training regime.

    :model (base.ModelType): The model to be trained.
    :cross_validate (int, default = None): The number of folds to use for cross validation.
    :grid_search (dict, default = None): The parameters to search over.
    """
    if grid_search is not None:
        train_sklearn_gridsearch_model(model, cross_validate = cross_validate, grid_search = grid_search, **kwargs)
    elif cross_validate is not None:
        train_sklearn_cv_model(model, cross_validate = cross_validate, **kwargs)
    else:
        train_sklearn_model(**kwargs)
