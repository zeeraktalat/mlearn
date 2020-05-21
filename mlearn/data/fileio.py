import sys
import ast
import json
from mlearn import base
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
            breakpoint()

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
    __import__('pdb').set_trace()
    for i in range(epochs):
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
