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


def write_predictions(pred_fn: base.Callable, data: base.DataType, dataset: GeneralDataset, model: base.ModelType,
                      model_header: list, train_field: str, label_field: str, data_name: str, main_name: str,
                      **kwargs) -> None:
    """
    Write documents and their predictions along with info about model making the prediction.

    :pred_fn (base.Callable): Opened result-file.
    :data (base.DataType): The data that were predicted on.
    :dataset (GeneralDataset): The dataset object.
    :model (base.ModelType): The model used for inference.
    :model_header (list): List of parameters in output file.
    :train_field (str): Attribute that is predicted on.
    :label_field (str): Attribute in data that contains the label.
    :model_info (list): Model information
    :data_name (str): Dataset evaluated on.
    :main_name (str): Dataset trained on.
    """
    model_info = [model.params.get(field, '-') for field in model_header]
    base = [_get_datestr, main_name, data_name]
    for doc in data:
        parsed = " ".join(getattr(doc, train_field)).replace('\n', ' ').replace('\r', ' ')
        pred_info = [doc.original, parsed,
                     dataset.label_ix_lookup(getattr(doc, label_field)), dataset.label_ix_lookup(doc.pred)]
        out = base + pred_info + model_info
        pred_fn.writerow(out)
    pred_fn.writerow(len(out) * ['---'])


def write_results(writer: base.Callable, model: base.ModelType, model_header: list, metric_header: list,
                  train_scores: object, dev_scores: object = None, data_name: str, main_name: str, **kwargs) -> None:
    """
    Write results to file.

    :writer (base.Callable): Path to file.
    :model (base.ModelType): The model to be written for.
    :model_header (list): Model parameters in the order they appear in the file.
    :metric_header (list): Metrics in the order they appear in the output file.
    :train_scores (dict): Train scores.
    :dev_scores (dict): Dev scores.
    :data_name (str): Name of the dataset that's being run on.
    :main_name (str): Name of the dataset the model is trained/being trained on.
    """
    base = [_get_datestr(), main_name, data_name]
    for i in range(len(train_score)):
        info = [model.params.get(field, '-') for field in model_header]
        results = [train_scores.scores.get(score, (i + 1) * ['-'])[i] for score in metric_header]

        if dev_scores:
            dev_results = [dev_scores.scores.get(score, (i + 1) * ['-'])[i] for score in metric_header]
            results.extend(dev_results)

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
