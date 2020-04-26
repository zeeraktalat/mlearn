import torch
import torch.nn as nn
from mlearn import base
import torch.nn.functional as F


class EmbeddingLSTMClassifier(nn.Module):
    """Embedding based LSTM classifier."""

    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, output_dim: int, num_layers: int,
                 dropout: float = 0.2, batch_first: bool = True, **kwargs) -> None:
        """
        Initialise the LSTM.

        :input_dim (int): The dimensionality of the input to the embedding generation.
        :hidden_dim (int): The dimensionality of the hidden dimension.
        :output_dim (int): Number of classes for to predict on.
        :num_layers (int): The number of recurrent layers in the LSTM (1-3).
        :dropout (float, default = 0.2): The strength of the dropout as a float [0.0;1.0]
        :batch_first (bool, default = True): Batch the first dimension?
        """
        super(EmbeddingLSTMClassifier, self).__init__()
        self.batch_first = batch_first
        self.name = 'lstm'

        self.itoe = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first = batch_first)
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
        out = self.dropout(self.itoe(sequence))
        out, last_layer = self.lstm(out)
        out = self.htoo(last_layer[0])
        prob_dist = self.softmax(out)

        return prob_dist.squeeze(0)


class EmbeddingMLPClassifier(nn.Module):
    """Embedding based MLP Classifier."""

    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2,
                 batch_first: bool = True, activation: str = 'tanh', **kwargs) -> None:
        """
        Initialise the model.

        :input_dim: The dimension of the input to the model.
        :hidden_dim: The dimension of the hidden layer.
        :output_dim: The dimension of the output layer (i.e. the number of classes).
        :batch_first (bool): Batch the first dimension?
        :activation (str, default = 'tanh'): String name of activation function to be used.
        """
        super(EmbeddingMLPClassifier, self).__init__()
        self.batch_first = batch_first
        self.name = 'mlp'

        self.itoe = nn.Embedding(input_dim, embedding_dim)
        self.htoh = nn.Linear(embedding_dim, hidden_dim)
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
        dropout = self.dropout if self.mode else lambda x: x
        out = dropout(self.activation(self.itoe(sequence)))
        out = dropout(self.activation(self.htoh(out)))
        out = out.mean(0)
        out = self.htoo(out)
        prob_dist = self.softmax(out)  # Re-shape to fit batch size.

        return prob_dist


class CNNClassifier(nn.Module):
    """CNN Classifier."""

    def __init__(self, window_sizes: base.List[int], num_filters: int, max_feats: int, hidden_dim: int, output_dim: int,
                 batch_first: bool = True, **kwargs) -> None:
        """
        Initialise the model.

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

        self.itoh = nn.Embedding(max_feats, hidden_dim)  # Works
        self.conv = nn.ModuleList([nn.Conv2d(1, num_filters, (w, hidden_dim)) for w in window_sizes])
        self.linear = nn.Linear(len(window_sizes) * num_filters, output_dim)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, sequence) -> base.DataType:
        """
        Forward step of the model.

        :sequence: The sequence to be predicted on.
        :return (base.DataType): The scores computed by the model.
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


class EmbeddingRNNClassifier(nn.Module):
    """Embedding RNN Classifier."""

    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2,
                 batch_first: bool = True, **kwargs) -> None:
        """
        Initialise the RNN classifier.

        :input_dim (int): The dimension of the input to the network.
        :embdding_dim (int): The dimension of the embeddings.
        :hidden_dim (int): The dimension of the hidden representation.
        :output_dim (int): The dimension of the output representation.
        :dropout (float, default = 0.2): The strength of the dropout [0.0; 1.0]
        :batch_first (bool): Is batch the first dimension?
        """
        super(EmbeddingRNNClassifier, self).__init__()
        self.batch_first = batch_first
        self.name = 'rnn'

        # Initialise the hidden dim
        self.hidden_dim = hidden_dim

        # Define layers of the network
        self.itoe = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first = batch_first)
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
        hidden = self.itoe(sequence)  # Map from input to hidden representation
        hidden, last_h = self.rnn(hidden)
        output = self.htoo(last_h)  # Map from hidden representation to output
        prob_dist = self.softmax(output)  # Generate probability distribution of output

        return prob_dist.squeeze(0)