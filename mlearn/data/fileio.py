import ast
import json
import torch
import joblib
from mlearn import base
from mlearn.data.dataset import GeneralDataset
from mlearn.utils.pipeline import get_deep_dict_value, _get_datestr


def read_json(fh: str, enc, doc_key: str, label_key: str, secondary_keys: dict = None, **kwargs) -> base.Tuple[str]:
    """
    Read JSON file containing entire document and label.

    To access keys in nested dictionaries, use the syntax <outer_key>.<inner_key>. Max depth 4.

    :fh: Filename
    :enc: Encoding of the file.
    :doc_key: Key to access document.
    :label_key: Key to access label.
    :secondary_keys (dict): Other keys to retrieve, including second level keys.
    :kwargs: Other keys to access.
    :return (tuple): Document, label, and other indexed items.
    """
    with open(fh, 'r', encoding = enc) as inf:
        for line in inf:
            try:
                dict_in = json.loads(line)
            except Exception as e:
                print("Error occurred: {0}".format(e))
                dict_in = ast.literal_eval(line)
            out_dict = {doc_key: dict_in.get(doc_key), label_key: dict_in.get(label_key)}

            if secondary_keys is not None:
                for key, vals in secondary_keys.items():
                    out_dict.update({key: get_deep_dict_value(dict_in, vals, None)})
            yield out_dict


def write_predictions(writer: base.Callable, model: base.ModelType, model_hdr: list, data_name: str, main_name: str,
                      hyper_info: list, data: base.DataType, dataset: GeneralDataset, train_field: str,
                      label_field: str, **kwargs) -> None:
    """
    Write documents and their predictions along with info about model making the prediction.

    :writer (base.Callable): Opened result-file.
    :model (base.ModelType): The model used for inference.
    :model_hdr (list): List of parameters in output file.
    :data_name (str): Dataset evaluated on.
    :main_name (str): Dataset trained on.
    :hyper_info (list): List of hyper parameters.
    :data (base.DataType): The data that were predicted on.
    :dataset (GeneralDataset): The dataset object to transform integer labels into strings.
    :train_field (str): Attribute that is predicted on.
    :label_field (str): Attribute in data that contains the label.
    """
    base = [_get_datestr(), main_name, data_name] + hyper_info
    info = [model.info.get(field, '-') for field in model_hdr]

    for doc in data:
        parsed = " ".join(getattr(doc, train_field)).replace('\n', ' ').replace('\r', ' ')
        label = dataset.label_ix_lookup(getattr(doc, label_field))
        pred = dataset.label_ix_lookup(doc.pred)

        pred_info = [label, pred, doc.original, parsed]

        out = base + info + pred_info
        writer.writerow(out)
    return True


def write_results(writer: base.Callable, model: base.ModelType, model_hdr: list, data_name: str, main_name: str,
                  hyper_info: list, metric_hdr: list, metrics: object, dev_metrics: object = None, **kwargs
                  ) -> None:
    """
    Write results to file.

    :writer (base.Callable): Path to file.
    :model (base.ModelType): The model to be written for.
    :model_hdr (list): Model parameters in the order they appear in the file.
    :data_name (str): Name of the dataset that's being run on.
    :main_name (str): Name of the dataset the model is trained/being trained on.
    :hyper_info (list): List of hyper-parameters.
    :metric_hdr (list): Metrics in the order they appear in the output file.
    :metrics (dict): Train scores.
    :dev_metrics (dict): dev_metrics scores.
    """
    base = [_get_datestr(), main_name, data_name] + hyper_info
    info = [model.info.get(field, '-') for field in model_hdr]

    for i in range(len(metrics['loss'])):
        results = [metrics.scores.get(score, (i + 1) * ['-'])[i] for score in metric_hdr]

        if dev_metrics:
            dev_metrics_results = [dev_metrics.scores.get(score, (i + 1) * ['-'])[i] for score in metric_hdr]
            results.extend(dev_metrics_results)

        out = base + info + results
        writer.writerow(out)
    return True


def mtl_batch_writer(writer: base.Callable, model: base.ModelType, model_hdr: list, task_name: str, main_name: str,
                     hyper_info: list, metric_hdr: list, metrics: object, epoch: int, batch: int, **kwargs) -> None:
    """
    Write results to file.

    :writer (base.Callable): Path to file.
    :model (base.ModelType): The model to be written for.
    :model_hdr (list): Model parameters in the order they appear in the file.
    :task_name (str): Name of the task/dataset that's being run on.
    :main_name (str): Name of the main task/dataset.
    :hyper_info (list): List of hyper-parameter values.
    :metric_hdr (list): Metrics in the order they appear in the output file.
    :metrics (dict): Train scores.
    :epoch (int): The iteration of the epoch.
    :batch (int): The iteration of the batches.
    """

    # Timestamp, Epoch, batch, Task, Main task
    base = [_get_datestr(), epoch, batch, task_name, main_name]

    # Modelinfo: model, input, embedding, hidden, output, dropout, nonlinear
    model_info = [model.info.get(field, '-') for field in model_hdr]

    # metrics + loss
    scores = [metrics.get_last(m) for m in metric_hdr]

    # hyper-info: batch size, # Epochs, Learning rate
    out = base + hyper_info + model_info + scores
    writer.writerow(out)


def store_model(model: base.ModelType, base_path: str, library: str = None) -> None:
    """
    Store model.

    :model (base.ModelType): The model to store.
    :base_path (str): Path to store the model in.
    :library (str, default = None)
    """
    if library is None:
        torch.save(model.state_dict(), f'{base_path}_{model.name}.mdl')
    else:
        joblib.dump(model.model, f'{base_path}_{model.name}.mdl')
        joblib.dump(model.vect, f'{base_path}_{model.name}.vct')


def load_model(model: base.ModelType, base_path: str, library: str = None) -> base.ModelType:
    """
    Load model.

    :model (base.ModelType): The model to store.
    :base_path (str): Path to load the model from.
    :library (str, default = None)
    """
    if library is None:
        return torch.load_statedict(torch.load(f'{base_path}_{model.name}.mdl'))
    else:
        return joblib.load(f'{base_path}.mdl'), joblib.load(f'{base_path}.vct')


def store_features(features: dict, base_path: str):
    """
    Store features.

    :feaures (dict): The feature dict to store.
    :base_path (str): Path to store the model in.
    """
    joblib.dump(features, f'{base_path}.fts')


def load_features(base_path) -> dict:
    """
    Load features.

    :base_path (str): Path to store the model in.
    """
    return joblib.load(f'{base_path}.fts')
