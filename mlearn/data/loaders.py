from mlearn import base
from mlearn.data.dataset import GeneralDataset


def _loader(args: dict) -> GeneralDataset:
    """
    Load the dataset.

    :args (dict): Dict containing arguments to load dataaset.
    :returns (GeneralDataset): Loaded and splitted dataset.
    """
    dataset = GeneralDataset(**args)
    dataset.load('train')

    if (args['dev'], args['test']) == (None, None):
        dataset.split(dataset.data, [0.8, 0.1, 0.1])

    elif args['dev'] is not None and args['test'] is None:
        dataset.load('dev')
        dataset.split(dataset.data, 0.8)

    elif args['dev'] is None and args['test'] is not None:
        dataset.split(dataset.data, 0.8)
        dataset.dev_set = dataset.test
        dataset.load('test')

    else:
        dataset.load('dev')
        dataset.load('test')

    return dataset


def davidson_to_binary(label: str) -> str:
    """
    Convert Davidson labels to binary labels.

    :label: Raw label as string
    :returns (str): label as str.
    """
    if label in ['0', '1']:
        return 'abuse'
    else:
        return 'not-abuse'


def davidson(cleaners: base.Callable, data_path: str, length: int = None, preprocessor: base.Callable = None,
             transformer: base.Callable = None, label_processor: base.Callable = None) -> GeneralDataset:
    """
    Load the davidson dataset.

    :cleaners (base.Callable): Initialized cleaner.
    :data_path (str): Path to data files.
    :length (int, default = None): Maximum length of sequence.
    :preprocessor (base.Callable, default = None): Preprocessor allowing for different experiments.
    :transformer (base.Callable, default = None): Document processing, if additional processing is required.
    :label_preprocessor (base.Callable, default = None): Label preprocessing, allowing for modifying the labelset.
    :returns (GeneralDataset): Loaded datasets.
    """
    args = {'data_dir': data_path,
            'ftype': 'csv',
            'fields': None,
            'train': 'davidson_offensive.csv', 'dev': None, 'test': None,
            'train_labels': None, 'dev_labels': None, 'test_labels': None,
            'sep': ',',
            'tokenizer': cleaners.tokenize,
            'preprocessor': preprocessor,
            'transformations': transformer,
            'length': length,
            'label_preprocessor': label_processor,
            'name': 'Davidson et al.'
            }

    ignore = base.Field('ignore', train = False, label = False, ignore = True)
    d_text = base.Field('text', train = True, label = False, ignore = False, ix = 6, cname = 'text')
    d_label = base.Field('label', train = False, label = True, cname = 'label', ignore = False, ix = 5)

    args['fields'] = [ignore, ignore, ignore, ignore, ignore, d_label, d_text]

    return _loader(args)


def waseem_to_binary(label: str) -> str:
    """
    Turn Waseem labels into binary labels.

    :label: String as label.
    :returns (str): label
    """
    if label.lower() in ['sexism', 'racism', 'both']:
        return 'abuse'
    else:
        return 'not-abuse'


def waseem(cleaners: base.Callable, data_path: str, length: int = None, preprocessor: base.Callable = None,
           transformer: base.Callable = None, label_processor: base.Callable = None) -> GeneralDataset:
    """
    Load the Waseem dataset (expert annotations).

    :cleaners (base.Callable): Initialized cleaner.
    :data_path (str): Path to data directory.
    :length (int), default = None): Maximum length of sequence.
    :preprocessor (base.Callable, default = None): Preprocessor allowing for different experiments.
    :transformer (base.Callable, default = None): Additional document processing, if required.
    :label_processor (base.Callable, default = None): Label preprocessing, allowing for modifying labelset.
    :returns (GeneralDataset): Loaded datasets.
    """
    args = {'data_dir': data_path,
            'ftype': 'json',
            'fields': None,
            'train': 'Wamateur_expert.json', 'dev': None, 'test': None,
            'train_labels': None, 'dev_labels': None, 'test_labels': None,
            'sep': None,
            'tokenizer': cleaners.tokenize,
            'preprocessor': preprocessor,
            'transformations': transformer,
            'length': length,
            'label_preprocessor': label_processor,
            'name': 'Waseem'
            }

    text_field = base.Field('text', train = True, label = False, ignore = False, cname = 'text')
    label_field = base.Field('label', train = False, label = True, ignore = False, cname = 'Annotation')
    args['fields'] = [text_field, label_field]

    return _loader(args)


def waseem_hovy(cleaners: base.Callable, data_path: str, train: str, length: int = None,
                preprocessor: base.Callable = None, transformer: base.Callable = None,
                label_processor: base.Callable = None) -> GeneralDataset:
    """
    Load the Waseem-Hovy dataset.

    :cleaners (base.Callable): Initialized cleaner.
    :data_path (str): Path to data directory.
    :length (int), default = None): Maximum length of sequence.
    :preprocessor (base.Callable, default = None): Preprocessor allowing for different experiments.
    :transformer (base.Callable, default = None): Additional document processing, if required.
    :returns (GeneralDataset): Loaded datasets.
    """
    args = {'data_dir': data_path,
            'ftype': 'json',
            'fields': None,
            'train': 'waseem_hovy.json', 'dev': None, 'test': None,
            'train_labels': None, 'dev_labels': None, 'test_labels': None,
            'sep': None,
            'tokenizer': cleaners.tokenize,
            'preprocessor': preprocessor,
            'transformations': transformer,
            'length': length,
            'label_preprocessor': label_processor,
            'name': 'Waseem-Hovy'
            }

    text_field = base.Field('text', train = True, label = False, ignore = False, cname = 'text')
    label_field = base.Field('label', train = False, label = True, ignore = False, cname = 'Annotation')
    args['fields'] = [text_field, label_field]

    return _loader(args)


def binarize_garcia(label: str) -> str:
    """
    Streamline Garcia labels with the other datasets.

    :returns (str): streamlined labels.
    """
    if label == 'hate':
        return 'abuse'
    else:
        return 'not-abuse'


def garcia(cleaners: base.Callable, data_path: str, length: int = None,
           preprocessor: base.Callable = None, transformer: base.Callable = None,
           label_processor: base.Callable = None) -> GeneralDataset:
    """
    Load the Garcia et al. dataset.

    :cleaners (base.Callable): Initialized cleaner.
    :data_path (str): Path to data directory.
    :length (int), default = None): Maximum length of sequence.
    :preprocessor (base.Callable, default = None): Preprocessor allowing for different experiments.
    :transformer (base.Callable, default = None): Additional document processing, if required.
    :label_processor (base.Callable, default = None): Label preprocessing, allowing for modifying labelset.
    :returns (GeneralDataset): Loaded datasets.
    """
    args = {'data_dir': data_path,
            'ftype': 'tsv',
            'fields': None,
            'train': 'garcia_stormfront_train.tsv', 'dev': None, 'test': 'garcia_stormfront_test.tsv',
            'train_labels': None, 'dev_labels': None, 'test_labels': None,
            'sep': '\t',
            'tokenizer': cleaners.tokenize,
            'preprocessor': preprocessor,
            'transformations': transformer,
            'length': length,
            'label_preprocessor': label_processor,
            'name': 'Garcia et al.'
            }

    text_field = base.Field('text', train = True, label = False, ignore = False, cname = 'text', ix = 5)
    label_field = base.Field('label', train = False, label = True, ignore = False, cname = 'label', ix = 4)
    id_field = base.Field('idx', train = False, label = False, ignore = False)
    user_field = base.Field('user_idx', train = False, label = False, ignore = False, cname = 'user_id')
    ignore = base.Field('ignore', train = False, label = False, ignore = True)

    args['fields'] = [id_field, user_field, ignore, ignore, label_field, text_field]

    return _loader(args)


def wulczyn(cleaners: base.Callable, data_path: str, length: int = None, preprocessor: base.Callable = None,
            transformer: base.Callable = None, label_processor: base.Callable = None) -> GeneralDataset:
    """
    Load the Wulczyn et al. dataset.

    :cleaners (base.Callable): Initialized cleaner.
    :data_path (str): Path to data directory.
    :length (int), default = None): Maximum length of sequence.
    :preprocessor (base.Callable, default = None): Preprocessor allowing for different experiments.
    :transformer (base.Callable, default = None): Additional document processing, if required.
    :label_processor (base.Callable, default = None): Label preprocessing, allowing for modifying labelset.
    :returns (GeneralDataset): Loaded datasets.
    """
    # Labelfield needs to be set to nothing, then fields need to be modified
    args = {'data_dir': data_path,
            'ftype': 'tsv',
            'fields': None,
            'train': 'wulczyn_train.tsv', 'dev': 'wulczyn_dev.tsv', 'test': 'wulczyn_test.tsv',
            'train_labels': None, 'dev_labels': None, 'test_labels': None,
            'sep': '\t',
            'tokenizer': cleaners.tokenize,
            'preprocessor': preprocessor,
            'transformations': transformer,
            'length': length,
            'label_preprocessor': label_processor,
            'name': 'Wulczyn et al.'
            }

    text = base.Field('text', train = True, label = False, cname = 'comment', ix = 1)
    label = base.Field('label', train = False, label = True, cname = 'label', ix = 2)
    idx = base.Field('id', train = False, label = False)
    ignore = base.Field('ignore', train = False, label = False, ignore = True)

    args['fields'] = [idx, text, label, ignore]

    return _loader(args)


def hoover(cleaners: base.Callable, data_path: str, length: int = None, preprocessor: base.Callable = None,
           transformer: base.Callable = None, label_processor: base.Callable = None) -> GeneralDataset:
    """
    Load the Hoover et al. dataset.

    :cleaners (base.Callable): Initialized cleaner.
    :data_path (str): Path to data directory.
    :length (int), default = None): Maximum length of sequence.
    :preprocessor (base.Callable, default = None): Preprocessor allowing for different experiments.
    :transformer (base.Callable, default = None): Additional document processing, if required.
    :label_processor (base.Callable, default = None): Label preprocessing, allowing for modifying labelset.
    :returns: Loaded datasets.
    """
    args = {'data_dir': data_path,
            'ftype': 'tsv',
            'fields': None,
            'train': 'MFTC_V4_text_parsed.tsv', 'dev': None, 'test': None,
            'sep': '\t',
            'tokenizer': cleaners.tokenize,
            'preprocessor': preprocessor,
            'transformations': transformer,
            'length': length,
            'label_preprocessor': label_processor,
            'name': 'Hoover et al.'}

    text = base.Field('text', train = True, label = False, cname = 'text', ix = 1)
    label = base.Field('label', train = False, label = True, cname = 'label', ix = 18)
    ignore = base.Field('ignore', train = False, label = False, cname = 'ignore', ignore = True)

    args['fields'] = [ignore, text] + 16 * [ignore] + [label, ignore]

    return _loader(args)


def guest_to_multiclass(label: str) -> str:
    """
    Map Gues labels to multiclass.

    :label (str): Raw label.
    :return (str): Mapped label.
    """
    if label == 'entity_directed_hostility':
        return label
    elif label == 'counter_speech':
        return 'discussion_of_eastasian_prejudice'
    elif label == 'discussion_of_eastasian_prejudice':
        return label
    elif label == 'entity_directed_criticism':
        return label
    else:
        return 'negative'


def guest(cleaners: base.Callable, data_path: str, length: int = None, preprocessor: base.Callable = None,
          transformer: base.Callable = None, label_processor: base.Callable = None) -> GeneralDataset:
    """
    Load the Vidgen et al. dataset.

    :cleaners (base.Callable): Initialized cleaner.
    :data_path (str): Path to data directory.
    :length (int), default = None): Maximum length of sequence.
    :preprocessor (base.Callable, default = None): Preprocessor allowing for different experiments.
    :transformer (base.Callable, default = None): Additional document processing, if required.
    :label_processor (base.Callable, default = None): Label preprocessing, allowing for modifying labelset.
    :returns: Loaded datasets.
    """
    args = {'data_dir': data_path,
            'ftype': 'tsv',
            'fields': None,
            'train': 'guest.csv', 'dev': None, 'test': None,
            'sep': ',',
            'tokenizer': cleaners.tokenize,
            'preprocessor': preprocessor,
            'transformations': transformer,
            'length': length,
            'label_preprocessor': label_processor,
            'name': 'Guest et al.'}

    text = base.Field('text', train = True, label = False, cname = 'text', ix = 3)
    label = base.Field('label', train = False, label = True, cname = 'label', ix = 3)
    ignore = base.Field('ignore', train = False, label = False, cname = 'ignore', ignore = True)

    args['fields'] = [ignore, text] + 16 * [ignore] + [label, ignore]

    return _loader(args)