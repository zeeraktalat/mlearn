import sys
import torch
import numpy as np
from mlearn import base
from mlearn.data import loaders
from mlearn.utils.metrics import Metrics
from mlearn.utils.pipeline import process_and_batch
from mlearn.modeling.embedding import MLPClassifier
from mlearn.data.clean import Cleaner, Preprocessors
from mlearn.utils.train import train_singletask_model
from torchtext.data import TabularDataset, Field, LabelField, BucketIterator


class TorchTextDefaultExtractor:
    """A class to get index-tensor batches from torchtext data object."""

    def __init__(self, datafield: str, labelfield: str, dataloader: base.DataType):
        """Initialize batch generator for torchtext."""
        self.data, self.df, self.lf = dataloader, datafield, labelfield

    def __len__(self):
        """Get length of the batches."""
        return len(self.data)

    def __iter__(self):
        """Iterate over batches in the data."""
        for batch in self.data:
            X = getattr(batch, self.df)
            y = getattr(batch, self.lf)
            yield (X, y)


# Initialize experiment
datadir = '../Generalisable_abuse/data/'
torch.random.manual_seed(42)
np.random.seed(42)
encoding = 'index'
tokenizer = 'bpe'
metrics = ['f1-score', 'precision', 'recall', 'accuracy']
display_metric = stop_metric = 'f1-score'
batch_size = 64
epochs = 5  # 50
learning_rate = 0.001
dropout = 0.0
embedding = 100
hidden = 100
nonlinearity = 'relu'
gpu = False
hyperopt = False
save_path = None
ts_train_metrics = Metrics(metrics, display_metric, stop_metric)
ts_dev_metrics = Metrics(metrics, display_metric, stop_metric)
mln_train_metrics = Metrics(metrics, display_metric, stop_metric)
mln_dev_metrics = Metrics(metrics, display_metric, stop_metric)

c = Cleaner(['url', 'hashtag', 'username', 'lower'])
experiment = Preprocessors(datadir).select_experiment('word')
onehot = True if encoding == 'onehot' else False

annotate=None
filters=[]
if tokenizer == 'spacy':
    tokenizer = c.tokenize
elif tokenizer == 'bpe':
    tokenizer = c.bpe_tokenize
elif tokenizer == 'ekphrasis':
    tokenizer = c.ekphrasis_tokenize
    annotate = {'elongated', 'emphasis'}
    filters = [f"<{filtr}>" for filtr in annotate]
    c._load_ekphrasis(annotate, filters)


ts_text = Field(tokenize=tokenizer, lower=True, batch_first=True)
ts_label = LabelField()
ts_fields = [('ignore', None), ('text', ts_text), ('label', ts_label), ('ignore', None)]
ts_train, ts_dev, ts_test = TabularDataset.splits('/home/cae/Code/Generalisable_abuse/data/', train='wulczyn_tiny_train.tsv',
                                                  validation='wulczyn_tiny_dev.tsv', test='wulczyn_tiny_test.tsv',
                                                  format='tsv', skip_header=True, fields=ts_fields)
ts_text.build_vocab(ts_train)
ts_label.build_vocab(ts_train)
ts_train_ds, ts_dev_ds = BucketIterator.splits(datasets=(ts_train, ts_dev), batch_size=batch_size)
print(f"TS train batch sizes, {len(ts_train_ds)}")
ts_batched_train = TorchTextDefaultExtractor('text', 'label', ts_train_ds)
ts_batched_dev = TorchTextDefaultExtractor('text', 'label', ts_dev_ds)
print(f"train {len(ts_train)}, dev {len(ts_dev)}, test {len(ts_test)}")
print(f"vocab size: {len(ts_text.vocab.stoi)}, label count: {len(ts_label.vocab.stoi)}")


mln_main = loaders.wulczyn_tiny(tokenizer, datadir, preprocessor=experiment, label_processor=None,
                           stratify='label', annotate=annotate, filters=filters, skip_header=True)
mln_main.build_token_vocab(mln_main.data)
mln_main.build_label_vocab(mln_main.data)
mln_batched_train = process_and_batch(mln_main, mln_main.data, batch_size, onehot=False, shuffle=True)
mln_batched_dev = process_and_batch(mln_main, mln_main.dev, batch_size, onehot=False, shuffle=True)
print(f"MLN train batch sizes, {len(mln_batched_train)}")
print(f"vocab size: {mln_main.vocab_size()}, label count: {mln_main.label_count()}")

print(f"different tokens: {set(mln_main.stoi.keys()) ^ set(ts_text.vocab.stoi.keys())}")
for token, ts_token_count in ts_text.vocab.freqs.items():
    mln_token_count = mln_main.token_counts.get(token)
    if ts_token_count != mln_token_count:
        print(f"different token count {token}: ts {ts_token_count} mln {mln_token_count}")

ts_model = MLPClassifier(len(ts_text.vocab.itos), embedding, hidden, len(ts_label.vocab.itos), dropout=False,
                         nonlinearity=nonlinearity)
ts_loss = torch.nn.NLLLoss()
ts_optimizer = torch.optim.Adam(ts_model.parameters(), learning_rate)
mln_model = MLPClassifier(mln_main.vocab_size(), embedding, hidden, mln_main.label_count(), dropout=False,
                          nonlinearity=nonlinearity)
mln_loss = torch.nn.NLLLoss()
mln_optimizer = torch.optim.Adam(mln_model.parameters(), learning_rate)

# ts side
print("####################### TENSORFLOW TRAIN #######################")
train_singletask_model(ts_model, save_path, epochs, ts_batched_train, ts_loss, ts_optimizer, ts_train_metrics,
                       dev=ts_batched_dev, dev_metrics=ts_dev_metrics, shuffle=False, gpu=True)
# mln side
print("####################### MLEARN TRAIN #######################")
train_singletask_model(mln_model, save_path, epochs, mln_batched_train, mln_loss, mln_optimizer, mln_train_metrics,
                       dev=mln_batched_dev, dev_metrics=mln_dev_metrics, shuffle=False, gpu=True)
