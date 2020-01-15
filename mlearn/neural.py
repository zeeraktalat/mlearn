import torch
import torch.nn as nn
from .shared import base
import torch.nn.functional as F


class LSTMClassifier(nn.Module):

    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, output_dim: int, num_layers: int,
                 batch_first: bool = True, **kwargs):
        """Initialise the LSTM.
        :input_dim (int): The dimensionality of the input to the embedding generation.
        :hidden_dim (int): The dimensionality of the hidden dimension.
        :output_dim (int): Number of classes for to predict on.
        :num_layers (int): The number of recurrent layers in the LSTM (1-3).
        :batch_first (bool): Batch the first dimension?
        """
        super(LSTMClassifier, self).__init__()
        self.batch_first = batch_first
        self.name = 'lstm'

        self.itoh = nn.Linear(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first = batch_first)
        self.htoo = nn.Linear(hidden_dim, output_dim)

        # Set the method for producing "probability" distribution.
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, sequence):
        """The forward step in the classifier.
        :sequence: The sequence to pass through the network.
        :return scores: The "probability" distribution for the classes.
        """
        if not self.batch_first:
            sequence = sequence.transpose(0, 1)

        sequence = sequence.float()
        out = self.itoh(sequence)  # Get embedding for the sequence
        out, last_layer = self.lstm(out)  # Get layers of the LSTM
        out = self.htoo(last_layer[0])
        prob_dist = self.softmax(out)  # The probability distribution

        return prob_dist.squeeze(0)

    @property
    def train_mode(self) -> bool:
        """Set or unset train mode.
        :mode (bool): True or False, setting train mode.
        :returns: Value of train mode.
        """
        return self.mode

    @train_mode.setter
    def train_mode(self, mode: bool) -> None:
        """Set train mode.
        :mode (bool): Bool value to set.
        """
        self.mode = mode


class MLPClassifier(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: int = 0.2, batch_first: bool = True,
                 **kwargs):
        """Initialise the model.
        :input_dim: The dimension of the input to the model.
        :hidden_dim: The dimension of the hidden layer.
        :output_dim: The dimension of the output layer (i.e. the number of classes).
        :batch_first (bool): Batch the first dimension?
        """
        super(MLPClassifier, self).__init__()
        self.batch_first = batch_first
        self.name = 'mlp'

        self.itoh = nn.Linear(input_dim, hidden_dim)
        self.htoh = nn.Linear(hidden_dim, hidden_dim)
        self.htoo = nn.Linear(hidden_dim, output_dim)

        # Set dropout and non-linearity
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, sequence):
        """The forward step in the classifier.
        :sequence: The sequence to pass through the network.
        :return scores: The "probability" distribution for the classes.
        """
        if self.batch_first:
            sequence = sequence.transpose(0, 1)

        sequence = sequence.float()
        dropout = self.dropout if self.mode else lambda x: x
        out = dropout(self.tanh(self.itoh(sequence)))
        out = dropout(self.tanh(self.htoh(out)))
        out = out.mean(0)
        out = self.htoo(out)
        prob_dist = self.softmax(out)  # Re-shape to fit batch size.

        return prob_dist

    @property
    def train_mode(self) -> bool:
        """Set or unset train mode.
        :mode (bool): True or False, setting train mode.
        :returns: Value of train mode.
        """
        return self.mode

    @train_mode.setter
    def train_mode(self, mode: bool) -> None:
        """Set train mode.
        :mode (bool): Bool value to set.
        """
        self.mode = mode


class CNNClassifier(nn.Module):

    def __init__(self, window_sizes: base.List[int], num_filters: int, max_feats: int, hidden_dim: int, output_dim: int,
                 batch_first: bool = True, **kwargs):
        """Initialise the model.
        :window_sizes: The size of the filters (e.g. 1: unigram, 2: bigram, etc.)
        :no_filters: The number of filters to apply.
        :max_feats: The maximum length of the sequence to consider.
        :hidden_dim (int): Hidden dimension size.
        :output_dim (int): Output dimension.
        :batch_first (bool, default: True): True if the batch is the first dimension.
        """
        super(CNNClassifier, self).__init__()
        self.batch_first = batch_first
        self.name = 'cnn'

        self.itoh = nn.Linear(max_feats, hidden_dim)  # Works
        self.conv = nn.ModuleList([nn.Conv2d(1, num_filters, (w, hidden_dim)) for w in window_sizes])
        self.linear = nn.Linear(len(window_sizes) * num_filters, output_dim)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, sequence):
        """The forward step of the model.
        :sequence: The sequence to be predicted on.
        :return scores: The scores computed by the model.
        """

        # CNNs expect batch first so let's try that
        if self.batch_first:
            sequence = sequence.transpose(0, 1)

        sequence = sequence.float()
        emb = self.itoh(sequence)  # Get embeddings for sequence
        output = [F.relu(conv(emb.unsqueeze(1))).squeeze(3) for conv in self.conv]
        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in output]
        output = torch.cat(output, 1)
        scores = self.softmax(self.linear(output))

        return scores

    @property
    def train_mode(self) -> bool:
        """Set or unset train mode.
        :mode (bool): True or False, setting train mode.
        :returns: Value of train mode.
        """
        return self.mode

    @train_mode.setter
    def train_mode(self, mode: bool) -> None:
        """Set train mode.
        :mode (bool): Bool value to set.
        """
        self.mode = mode


class RNNClassifier(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, batch_first: bool = True, **kwargs):
        """Initialise the RNN classifier.
        :input_dim: The dimension of the input to the network.
        :hidden_dim: The dimension of the hidden representation.
        :output_dim: The dimension of the output representation.
        :batch_first (bool): Is batch the first dimension?
        """
        super(RNNClassifier, self).__init__()
        self.batch_first = batch_first
        self.name = 'rnn'

        # Initialise the hidden dim
        self.hidden_dim = hidden_dim

        # Define layers of the network
        self.itoh = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first = batch_first)
        self.htoo = nn.Linear(hidden_dim, output_dim)

        # Set the method for producing "probability" distribution.
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sequence):
        """The forward step in the network.
        :inputs: The inputs to pass through network.
        :hidden: The hidden representation at the previous timestep.
        :return softmax, hidden: Return the "probability" distribution and the new hidden representation.
        """
        if not self.batch_first:
            sequence = sequence.transpose(0, 1)

        sequence = sequence.float()
        hidden = self.itoh(sequence)  # Map from input to hidden representation
        hidden, last_h = self.rnn(hidden)
        output = self.htoo(last_h)  # Map from hidden representation to output
        softmax = self.softmax(output)  # Generate probability distribution of output

        return softmax.squeeze(0)

    @property
    def train_mode(self) -> bool:
        """Set or unset train mode.
        :mode (bool): True or False, setting train mode.
        :returns: Value of train mode.
        """
        return self.mode

    @train_mode.setter
    def train_mode(self, mode: bool) -> None:
        """Set train mode.
        :mode (bool): Bool value to set.
        """
        self.mode = mode
