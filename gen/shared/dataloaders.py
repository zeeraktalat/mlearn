from . import base
from .data import GeneralDataset


def _loader(args: dict):
    """Loads the dataset.
    :args (dict): Dict containing arguments to load dataaset.
    :returns: Loaded and splitted dataset.
    """
    dataset = GeneralDataset(**args)
    dataset.load('train')

    if args['dev'] is not None:
        dataset.load('dev')
    if args['test'] is not None:
        dataset.load('test')
    if (args['dev'], args['test']) == (None, None):
        dataset.split(dataset.data, 0.8)

    return dataset


def davidson_to_binary(label: str) -> str:
    """TODO: Docstring for davidson_to_binary.
    :label: Raw label as string
    :returns: label as int.
    """
    if label in ['0', '1']:
        return 'abuse'
    else:
        return 'not-abuse'


def davidson(cleaners: base.Callable, preprocessor: base.Callable = None):
    """Function to load the davidson dataset.
    :cleaners (base.Callable): Initialized cleaner.
    :preprocessor (base.Callable, default = None): Preprocessor allowing for different experiments.
    :returns: Loaded datasets.
    """
    args = {'data_dir': '~/PhD/projects/active/Generalisable_abuse/trial/',
            'ftype': 'csv',
            'fields': None,
            'train': 'davidson_offensive.csv', 'dev': None, 'test': None,
            'train_labels': None, 'dev_labels': None, 'test_labels': None,
            'sep': ',',
            'tokenizer': cleaners.tokenize,
            'preprocessor': preprocessor,
            'transformations': None,
            'length': None,
            'label_preprocessor': davidson_to_binary,
            'name': 'Davidson et al.'
            }

    ignore = base.Field('ignore', train = False, label = False, ignore = True)
    d_text = base.Field('text', train = True, label = False, ignore = False, ix = 6, cname = 'text')
    d_label = base.Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 5)

    args['fields'] = [ignore, ignore, ignore, ignore, ignore, d_label, d_text]

    return _loader(args)


def waseem_to_binary(label: str) -> str:
    """Turn Waseem labels into binary labels.
    :label: String as label.
    :returns: label
    """
    if label.lower() in ['sexism', 'racism', 'both']:
        return 'abuse'
    else:
        return 'not-abuse'


def waseem(cleaners: base.Callable, preprocessor: base.Callable = None):
    """Load the Waseem dataset (expert annotations).
    :cleaners (base.Callable): Initialized cleaner.
    :preprocessor (base.Callable, default = None): Preprocessor allowing for different experiments.
    :returns: Loaded datasets.
    """
    args = {'data_dir': '~/PhD/projects/active/Generalisable_abuse/trial/',
            'ftype': 'json',
            'fields': None,
            'train': 'Wamateur_expert.json', 'dev': None, 'test': None,
            'train_labels': None, 'dev_labels': None, 'test_labels': None,
            'sep': None,
            'tokenizer': cleaners.tokenize,
            'preprocessor': preprocessor,
            'transformations': None,
            'length': None,
            'label_preprocessor': waseem_to_binary,
            'name': 'Waseem'
            }
    text_field = base.Field('text', train = True, label = False, ignore = False, cname = 'text')
    label_field = base.Field('label', train = False, label = True, ignore = False, cname = 'Annotation')
    args['fields'] = [text_field, label_field]

    return _loader(args)


def waseem_hovy(cleaners: base.Callable, preprocessor: base.Callable = None):
    """Load the Waseem-Hovy dataset.
    :cleaners (base.Callable): Initialized cleaner.
    :preprocessor (base.Callable, default = None): Preprocessor allowing for different experiments.
    :returns: Loaded datasets.
    """
    args = {'data_dir': '~/PhD/projects/active/Generalisable_abuse/trial/',
            'ftype': 'json',
            'fields': None,
            'train': 'waseem_hovy.json', 'dev': None, 'test': None,
            'train_labels': None, 'dev_labels': None, 'test_labels': None,
            'sep': None,
            'tokenizer': cleaners.tokenize,
            'preprocessor': preprocessor,
            'transformations': None,
            'length': None,
            'label_preprocessor': waseem_to_binary,
            'name': 'Waseem-Hovy'
            }
    text_field = base.Field('text', train = True, label = False, ignore = False, cname = 'text')
    label_field = base.Field('label', train = False, label = True, ignore = False, cname = 'Annotation')
    args['fields'] = [text_field, label_field]

    return _loader(args)


def streamline_garcia(label: str):
    """Streamline Garcia labels with the other datasets.
    :returns: streamlined labels.
    """
    if label == 'hate':
        return 'abuse'
    else:
        return 'not-abuse'


def garcia(cleaners: base.Callable, preprocessor: base.Callable = None):
    """Load the Garcia et al. dataset.
    :cleaners (base.Callable): Initialized cleaner.
    :preprocessor (base.Callable, default = None): Preprocessor allowing for different experiments.
    :returns: Loaded datasets.
    """
    args = {'data_dir': '~/PhD/projects/active/Generalisable_abuse/trial/',
            'ftype': 'tsv',
            'fields': None,
            'train': 'garcia_stormfront_train.tsv', 'dev': None, 'test': 'garcia_stormfront_test.tsv',
            'train_labels': None, 'dev_labels': None, 'test_labels': None,
            'sep': '\t',
            'tokenizer': cleaners.tokenize,
            'preprocessor': preprocessor,
            'transformations': None,
            'length': None,
            'label_preprocessor': streamline_garcia,
            'name': 'Garcia et al.'
            }

    text_field = base.Field('text', train = True, label = False, ignore = False, cname = 'text', ix = 5)
    label_field = base.Field('label', train = False, label = True, ignore = False, cname = 'label', ix = 4)
    id_field = base.Field('idx', train = False, label = False, ignore = False)
    user_field = base.Field('user_idx', train = False, label = False, ignore = False, cname = 'user_id')
    ignore = base.Field('ignore', train = False, label = False, ignore = True)

    args['fields'] = [id_field, user_field, ignore, ignore, label_field, text_field]

    return _loader(args)


def wulczyn(cleaners: base.Callable, preprocessor: base.Callable = None):
    """Load the Wulczyn et al. dataset.
    :cleaners (base.Callable): Initialized cleaner.
    :preprocessor (base.Callable, default = None): Preprocessor allowing for different experiments.
    :returns: Loaded datasets.
    """
    # Labelfield needs to be set to nothing, then fields need to be modified
    args = {'data_dir': '~/PhD/projects/active/Generalisable_abuse/trial/',
            'ftype': 'tsv',
            'fields': None,
            'train': 'wulczyn_train.tsv', 'dev': 'wulczyn_dev.tsv', 'test': 'wulczyn_test.tsv',
            'train_labels': None, 'dev_labels': None, 'test_labels': None,
            'sep': '\t',
            'tokenizer': cleaners.tokenize,
            'preprocessor': preprocessor,
            'transformations': None,
            'length': None,
            'label_preprocessor': None,
            'name': 'Wulczyn et al.'
            }

    text = base.Field('text', train = True, label = False, cname = 'comment', ix = 1)
    label = base.Field('label', train = False, label = True, cname = 'label', ix = 2)
    idx = base.Field('id', train = False, label = False)
    ignore = base.Field('ignore', train = False, label = False, ignore = True)

    args['fields'] = [idx, text, label, ignore]

    return _loader(args)
