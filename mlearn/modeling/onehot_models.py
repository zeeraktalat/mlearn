import torch
import torch.nn as nn
from mlearn import base
import torch.nn.functional as F


class OnehotLSTMClassifier(nn.Module):
    """Onehot LSTM classifier."""

    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, output_dim: int, num_layers: int,
                 batch_first: bool = True, **kwargs) -> None:
        """
        Initialise the LSTM.

        :input_dim (int): The dimensionality of the input to the embedding generation.
        :hidden_dim (int): The dimensionality of the hidden dimension.
        :output_dim (int): Number of classes for to predict on.
        :num_layers (int): The number of recurrent layers in the LSTM (1-3).
        :batch_first (bool, default = True): Batch the first dimension?
        """
        super(OnehotLSTMClassifier, self).__init__()
        self.batch_first = batch_first
        self.name = 'lstm'

        self.itoh = nn.Linear(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first = batch_first)
        self.htoo = nn.Linear(hidden_dim, output_dim)

        # Set the method for producing "probability" distribution.
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
        out, last_layer = self.lstm(out)  # Get layers of the LSTM
        out = self.htoo(last_layer[0])
        prob_dist = self.softmax(out)  # The probability distribution

        return prob_dist.squeeze(0)


class OnehotMLPClassifier(nn.Module):
    """Onehot MLP Classifier."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2, batch_first: bool = True,
                 activation: str = 'tanh', **kwargs) -> None:
        """
        Initialise the model.

        :input_dim: The dimension of the input to the model.
        :hidden_dim: The dimension of the hidden layer.
        :output_dim: The dimension of the output layer (i.e. the number of classes).
        :batch_first (bool): Batch the first dimension?
        :activation (str, default = 'tanh'): String name of activation function to be used.
        """
        super(OnehotMLPClassifier, self).__init__()
        self.batch_first = batch_first
        self.name = 'mlp'

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
        dropout = self.dropout if self.mode else lambda x: x
        out = dropout(self.activation(self.itoh(sequence)))
        out = dropout(self.activation(self.htoh(out)))
        out = out.mean(0)
        out = self.htoo(out)
        prob_dist = self.softmax(out)  # Re-shape to fit batch size.

        return prob_dist


class OnehotCNNClassifier(nn.Module):
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
        super(OnehotCNNClassifier, self).__init__()
        self.batch_first = batch_first
        self.name = 'cnn'

        self.itoh = nn.Linear(max_feats, hidden_dim)  # Works
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


class OnehotRNNClassifier(nn.Module):
    """Onehot RNN Classifier."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, batch_first: bool = True,
                 **kwargs) -> None:
        """
        Initialise the RNN classifier.

        :input_dim: The dimension of the input to the network.
        :hidden_dim: The dimension of the hidden representation.
        :output_dim: The dimension of the output representation.
        :batch_first (bool): Is batch the first dimension?
        """
        super(OnehotRNNClassifier, self).__init__()
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
        output = self.htoo(last_h)  # Map from hidden representation to output
        prob_dist = self.softmax(output)  # Generate probability distribution of output

        return prob_dist.squeeze(0)


class OnehotMTLLSTMClassifier(nn.Module):
    """Multitask LSTM Classifier."""

    def __init__(self, input_dims: base.List[int], shared_dim: int, hidden_dims: base.List[int],
                 output_dims: base.List[int], no_layers: int = 1, dropout: float = 0.2, batch_first = True,
                 **kwargs) -> None:
        """
        Initialise the Multitask LSTM.

        :param input_dims (base.List[int]): The dimensionality of the input.
        :param shared_dim (int): The dimensionality of the shared layers.
        :param hidden_dim (base.List[int]): The dimensionality of the hidden dimensions for each task.
        :param embedding_dim: The dimensionality of the the produced embeddings.
        :param no_classes: Number of classes for to predict on.
        :param no_layers: The number of recurrent layers in the LSTM (1-3).
        :param dropout: Value fo dropout
        """
        super(OnehotMTLLSTMClassifier, self).__init__()

        # Initialise the hidden dim
        self.all_parameters = nn.ParameterList()

        assert len(input_dims) == len(hidden_dims) == len(output_dims)

        # Input layer (not shared) [Linear]
        # hidden to hidden layer (shared) [Linear]
        # Hidden to hidden layer (not shared) [LSTM]
        # Output layer (not shared) [Linear}

        self.inputs = {}  # Define task inputs
        for task_id, input_dim in enumerate(input_dims):
            layer = nn.Linear(input_dim, hidden_dims[0])
            self.inputs[task_id] = layer

            # Add parameters
            self.all_parameters.append(layer.weight)
            self.all_parameters.append(layer.bias)

        self.shared = []
        for i in range(len(hidden_dims) - 1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.shared.append(layer)

            # Add parameters
            self.all_parameters.append(layer.weight)
            self.all_parameters.append(layer.bias)

        self.lstm = {}
        for task_ix, hdim in enumerate(hidden_dims):
            layer = nn.LSTM(hdim, hdim, batch_first = batch_first)  # Will go out of index.
            self.lstm[task_ix] = layer

            # Add parameters
            self.all_parameters.append(layer.weight_ih_l0)
            self.all_parameters.append(layer.weight_hh_l0)
            self.all_parameters.append(layer.bias_ih_l0)
            self.all_parameters.append(layer.bias_hh_l0)

        self.outputs = {}
        for task_id, _ in enumerate(hidden_dims):
            layer = nn.Linear(hidden_dims[task_id], output_dims[task_id])
            self.outputs[task_id] = layer

            # Add parameters
            self.all_parameters.append(layer.weight)
            self.all_parameters.append(layer.bias)

        # Set the method for producing "probability" distribution.
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim = 1)

        # TODO Ensure that the model is deterministic (the bias term is added)
        print(self)
        print(list(self.all_parameters))

    def forward(self, sequence, task_id) -> base.DataType:
        """
        Forward step in the classifier.

        :param sequence: The sequence to pass through the network.
        :param task_id: The task on which to perform forward pass.
        :return (base.DataType): The "probability" distribution for the classes.
        """
        res = self.dropout(self.inputs[task_id](sequence.float()))

        for layer in self.shared:
            res = self.dropout(layer(res))

        lstm_out, (lstm_hidden, _) = self.lstm[task_id](res)

        output = self.outputs[task_id](lstm_hidden)

        prob_dist = self.softmax(output)  # The probability distribution

        return prob_dist.squeeze(0)
