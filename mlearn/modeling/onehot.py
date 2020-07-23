import torch
import torch.nn as nn
from mlearn import base
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    """LSTM classifier."""

    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, output_dim: int, no_layers: int,
                 dropout: float = 0.0, batch_first: bool = True, **kwargs) -> None:
        """
        Initialise the LSTM.

        :input_dim (int): The dimensionality of the input to the embedding generation.
        :hidden_dim (int): The dimensionality of the hidden dimension.
        :output_dim (int): Number of classes for to predict on.
        :no_layers (int): The number of recurrent layers in the LSTM (1-3).
        :dropout (float, default = 0.0): Value of dropout layer.
        :batch_first (bool, default = True): Batch the first dimension?
        """
        super(LSTMClassifier, self).__init__()
        self.batch_first = batch_first
        self.name = 'lstm'
        self.info = {'Input dim': input_dim, 'Embedding dim': embedding_dim, 'Hidden dim': hidden_dim,
                     'Output dim': output_dim, '# layers': no_layers, 'Dropout': dropout, 'Model': self.name}

        self.itoh = nn.Linear(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, no_layers, batch_first = batch_first)
        self.htoo = nn.Linear(hidden_dim, output_dim)

        # Set the method for producing "probability" distribution.
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, sequence: base.DataType) -> base.DataType:
        """
        Forward step in the classifier.

        :sequence: The sequence to pass through the network.
        :return (base.DataType): The "probability" distribution for the classes.
        """
        if not self.batch_first:
            sequence = sequence.transpose(0, 1)

        sequence = sequence.float()
        out = self.itoh(sequence)  # Get embedding for the sequence
        out, (last_layer, _) = self.lstm(out)  # Get layers of the LSTM
        out = self.htoo(self.dropout(last_layer))
        prob_dist = self.softmax(out)  # The probability distribution

        return prob_dist.squeeze(0)


class MLPClassifier(nn.Module):
    """MLP Classifier."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0, activation: str = 'tanh',
                 batch_first: bool = True, **kwargs) -> None:
        """
        Initialise the model.

        :input_dim (int): The dimension of the input to the model.
        :hidden_dim (int): The dimension of the hidden layer.
        :output_dim (int): The dimension of the output layer (i.e. the number of classes).
        :dropout (float, default = 0.0): Value of dropout layer.
        :activation (str, default = 'tanh'): String name of activation function to be used.
        :batch_first (bool): Batch the first dimension?
        """
        super(MLPClassifier, self).__init__()
        self.batch_first = batch_first
        self.name = 'mlp'
        self.info = {'Model': self.name, 'Input dim': input_dim, 'Hidden dim': hidden_dim, 'Output dim': output_dim,
                     'Activation Func': activation, 'Dropout': dropout}

        self.itoh = nn.Linear(input_dim, hidden_dim)
        self.htoh = nn.Linear(hidden_dim, hidden_dim)
        self.htoo = nn.Linear(hidden_dim, output_dim)

        # Set dropout and non-linearity
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, sequence: base.DataType):
        """
        Forward step in the classifier.

        :sequence: The sequence to pass through the network.
        :return (base.DataType): The "probability" distribution for the classes.
        """
        if self.batch_first:
            sequence = sequence.transpose(0, 1)

        sequence = sequence.float()
        out = self.dropout(self.activation(self.itoh(sequence)))
        out = self.dropout(self.activation(self.htoh(out)))
        out = out.mean(0)
        out = self.htoo(out)
        prob_dist = self.softmax(out)  # Re-shape to fit batch size.

        return prob_dist


class CNNClassifier(nn.Module):
    """CNN Classifier."""

    def __init__(self, window_sizes: base.List[int], num_filters: int, input_dim: int, hidden_dim: int, output_dim: int,
                 activation: str = 'relu', batch_first: bool = True, **kwargs) -> None:
        """
        Initialise the model.

        :window_sizes (base.List[int]): The size of the filters (e.g. 1: unigram, 2: bigram, etc.)
        :no_filters (int): The number of filters to apply.
        :input_dim (int): The input dimension (can and should be limited beyond the raw input dimensions).
        :hidden_dim (int): Hidden dimension size.
        :output_dim (int): Output dimension.
        :activation (str): Name of activation function to use.
        :batch_first (bool, default: True): True if the batch is the first dimension.
        """
        super(CNNClassifier, self).__init__()
        self.batch_first = batch_first
        self.name = 'cnn'
        self.info = {'Model': self.name, 'Window Sizes': " ".join([str(it) for it in window_sizes]),
                     '# Filters': num_filters, 'Input dim': input_dim, 'Hidden dim': hidden_dim,
                     'Output dim': output_dim, 'Activation Func': activation}

        self.itoh = nn.Linear(input_dim, hidden_dim)  # Works
        self.conv = nn.ModuleList([nn.Conv2d(1, num_filters, (w, hidden_dim)) for w in window_sizes])
        self.linear = nn.Linear(len(window_sizes) * num_filters, output_dim)
        self.activation = F.relu if activation == 'relu' else F.tanh
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, sequence) -> base.DataType:
        """
        Forward step of the model.

        :sequence: The sequence to be predicted on.
        :return (base.DataType): The probability distribution computed by the model.
        """
        # CNNs expect batch first so let's try that
        if not self.batch_first:
            sequence = sequence.transpose(0, 1)

        sequence = sequence.float()
        emb = self.itoh(sequence)  # Get embeddings for sequence
        output = [self.activation(conv(emb.unsqueeze(1))).squeeze(3) for conv in self.conv]
        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in output]
        output = torch.cat(output, 1)
        prob_dist = self.softmax(self.linear(output))

        return prob_dist


class RNNClassifier(nn.Module):
    """RNN Classifier."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0, batch_first: bool = True,
                 **kwargs) -> None:
        """
        Initialise the RNN classifier.

        :input_dim (int): The dimension of the input to the network.
        :hidden_dim (int): The dimension of the hidden representation.
        :output_dim (int): The dimension of the output representation.
        :dropout (float, default = 0.0): The value of the dropout layer.
        :batch_first (bool): Is batch the first dimension?
        """
        super(RNNClassifier, self).__init__()
        self.batch_first = batch_first
        self.name = 'rnn'
        self.info = {'Model': self.name, 'Input dim': input_dim, 'Hidden dim': hidden_dim,
                     'Output dim': output_dim, 'Dropout': dropout}

        # Initialise the hidden dim
        self.hidden_dim = hidden_dim

        # Define layers of the network
        self.itoh = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first = batch_first)
        self.htoo = nn.Linear(hidden_dim, output_dim)

        # Set the method for producing "probability" distribution.
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sequence) -> base.DataType:
        """
        Forward step in the network.

        :inputs: The inputs to pass through network.
        :hidden: The hidden representation at the previous timestep.
        :return (base.DataType): Return the "probability" distribution and the new hidden representation.
        """
        if not self.batch_first:
            sequence = sequence.transpose(0, 1)

        sequence = sequence.float()
        hidden = self.itoh(sequence)  # Map from input to hidden representation
        hidden, last_h = self.rnn(hidden)
        output = self.htoo(self.dropout(last_h))  # Map from hidden representation to output)
        prob_dist = self.softmax(output)  # Generate probability distribution of output

        return prob_dist.squeeze(0)
