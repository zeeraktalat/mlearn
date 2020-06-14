import ast
import json
from mlearn import base
from mlearn.data.dataset import GeneralDataset
from mlearn.utils.pipeline import get_deep_dict_value


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


def write_predictions(data: base.DataType, dataset: GeneralDataset, train_field: str, label_field: str,
                      model_info: list, data_name: str, main_name: str, pred_fn: base.Callable,
                      **kwargs) -> None:
    """
    Write documents and their predictions along with info about model making the prediction.

    :data: The dataset objects that were predicted on.
    :train_field (str): Attribute that is predicted on.
    :label_field (str): Attribute in data that contains the label.
    :model_info (list): Model information
    :data_name (str): Dataset evaluated on.
    :main_name (str): Dataset trained on.
    :pred_fn (base.Callable): Opened resultfile.
    """
    for doc in data:
        try:
            out = [" ".join(getattr(doc, train_field)).replace('\n', ' ').replace('\r', ' '),
                   dataset.label_ix_lookup(getattr(doc, label_field)), dataset.label_ix_lookup(doc.pred),
                   data_name, main_name] + model_info
            pred_fn.writerow(out)
        except Exception:
            __import__('pdb').set_trace()

    pred_fn.writerow(len(out) * ['---'])


def write_results(writer: base.Callable, train_scores: dict, train_loss: list, dev_scores: dict, dev_loss: list,
                  epochs: int, model_info: list, metrics: list, exp_len: int, data_name: str, main_name: str,
                  **kwargs) -> None:
    """
    Write results to file.

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
    if isinstance(train_loss, float):
        train_loss = [train_loss]

    iterations = epochs if epochs == len(train_loss) else len(train_loss)
    for i in range(iterations):
        try:
            out = [data_name, main_name] + [i] + model_info  # Base info
            out += [train_scores[m][i] for m in metrics.list()] + [train_loss[i]]  # Train info

            if dev_scores:
                out += [dev_scores[m][i] for m in metrics.list()] + [dev_loss[i]]  # Dev info

        except IndexError:
            __import__('pdb').set_trace()

        row_len = len(out)
        if row_len < exp_len:
            out += [''] * (row_len - exp_len)
        elif row_len > exp_len:
            __import__('pdb').set_trace()

        writer.writerow(out)
