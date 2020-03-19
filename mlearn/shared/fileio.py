import sys
import ast
import json
from . import base


def read_json(fh: str, enc, doc_key: str, label_key: str, **kwargs) -> base.Tuple[str, str, ...]:
    """Read JSON file containing entire document and label.
    To access keys in nested dictionaries, use the syntax <outer_key>.<inner_key>. Max depth 4.
    :param fh: Filename
    :param enc: Encoding of the file.
    :param doc_key: Key to access document.
    :param label_key: Key to access label.
    :param kwargs: Other keys to access.
    :return data_tup: Tuple containing document, label, and other indexed items.
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


def write_results(performance: dict, parameters: dict, out_fh: str, mode: str) -> None:
    """Write results out to a tsv file.
    :param performance: Dictionary containing results of model.
    :param parameters: A dictionary containng the model parameters.
    :param out_fh: Filename of the output file.
    :param mode: Write mode of the file (options: ['a', 'w']).
    """
    fh = open('../../experiments/results/' + out_fh, mode = mode, encoding = 'utf-8')
    if mode == 'a':
        pass
    elif mode == 'w':
        first_line = "\t".join(performance.keys()) + "\t".join(parameters.keys())
        fh.write(first_line)

    # Write results
    output = "\t".join(performance.values()) + "\t".join(parameters.values())
    fh.write(output)

    fh.close()


def print_results(performance: dict, parameters: dict, iter_info: dict = {}, first: bool = False) -> None:
    """Print results in a readable manner.
    :param performance: Classifier metrics.
    :param parameters: Parameters for this iteration.
    :param iter_info: Information about the iteration.
    :param first: Whether it's the first line or not.
    """

    out = ""
    i_info = ""

    if iter_info:
        for k, v in iter_info.items():
            i_info = "{0}: {1} | ".format(k, v)
        i_info = i_info[0:-2] + ': '

    if first:
        out = "\t".join(performance.keys()) + ' | ' + "\t".join(parameters.keys())
        out = len(i_info) * " " + out
        print(out)

    out = "\t".join(performance.values()) + ' | ' + "\t".join(parameters.values())

    if iter_info != {}:
        out = i_info + out

    print(out)
