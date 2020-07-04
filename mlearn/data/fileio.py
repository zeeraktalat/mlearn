import sys
import ast
import json
import torch
import joblib
from mlearn import base
from mlearn.utils.pipeline import _get_datestr
from mlearn.data.dataset import GeneralDataset


def read_json(fh: str, enc, doc_key: str, label_key: str, **kwargs) -> base.Tuple[str]:
    """
    Read JSON file containing entire document and label.

    To access keys in nested dictionaries, use the syntax <outer_key>.<inner_key>. Max depth 4.

    :param fh: Filename
    :param enc: Encoding of the file.
    :param doc_key: Key to access document.
    :param label_key: Key to access label.
    :param kwargs: Other keys to access.
    :return (tuple): Document, label, and other indexed items.
    """
    for line in open('../../data/' + fh, 'r', encoding = enc):
        try:
            dict_in = json.loads(line)
        except Exception as e:
            print("Error occurred: {0}".format(e))
            dict_in = ast.literal_eval(line)
        finally:
            out_vals = []
            out_vals.append(dict_in[doc_key])
            out_vals.append(dict_in[label_key])

            if kwargs:  # Get all values in kwargs
                for key, val in kwargs.items():
                    keys = val.split('.')
                    key_count = len(keys)

                    try:
                        if key_count == 1:
                            out_vals.append(dict_in[val])
                        elif key_count == 2:
                            out_vals.append(dict_in[keys[0][keys[1]]])
                        elif key_count == 3:
                            out_vals.append(dict_in[keys[0]][keys[1]][keys[2]])
                        elif key_count == 4:
                            out_vals.append(dict_in[keys[0]][keys[1]][keys[2]][keys[3]])
                    except IndexError as e:
                        print("One of the keys does not exist.\nError {0}.\nkeys: {1}\nDoc: {2}".
                              format(e, keys, dict_in), file = sys.stderr)

    return tuple(out_vals)


def write_predictions(writer: base.Callable, model: base.ModelType, model_hdr: list, data_name: str, main_name: str,
                      hyper_info: str, data: base.DataType, dataset: GeneralDataset, train_field: str, label_field: str,
                      **kwargs) -> None:
    """
    Write documents and their predictions along with info about model making the prediction.

    :writer (base.Callable): Opened result-file.
    :model (base.ModelType): The model used for inference.
    :model_hdr (list): List of parameters in output file.
    :data_name (str): Dataset evaluated on.
    :main_name (str): Dataset trained on.
    :hyper_info (list): List of hyper paraemeters.
    :data (base.DataType): The data that were predicted on.
    :dataset (GeneralDataset): The dataset object.
    :train_field (str): Attribute that is predicted on.
    :label_field (str): Attribute in data that contains the label.
    """
    base = [_get_datestr(), main_name, data_name]
    info = [model.info.get(field, '-') for field in model_hdr] + hyper_info

    for doc in data:
        parsed = " ".join(getattr(doc, train_field)).replace('\n', ' ').replace('\r', ' ')
        label = dataset.label_ix_lookup(getattr(doc, label_field))
        pred = dataset.label_ix_lookup(doc.pred)

        pred_info = [doc.original, parsed, label, pred]

        out = base + pred_info + info
        writer.writerow(out)


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

    for i in range(len(metrics.scores['loss'])):
        results = [metrics.scores.get(score, (i + 1) * ['-'])[i] for score in metrics.scores]

        for score in metric_hdr:
            results.append(metrics.scores.get(score, (i + 1) * ['-'])[i])

        if dev_metrics:
            dev_metrics_results = [dev_metrics.scores.get(score, (i + 1) * ['-'])[i] for score in metric_hdr]
            results.extend(dev_metrics_results)

        out = base + info + results
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
