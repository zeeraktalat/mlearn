import torch
import torch.nn as nn
from mlearn import base


class EmbeddingLSTMClassifier(nn.Module):
    """Multitask LSTM Classifier."""

    def __init__(self, input_dims: base.List[int], embedding_dims: int, shared_dim: int, hidden_dims: base.List[int],
                 output_dims: base.List[int], no_layers: int = 1, dropout: float = 0.0, batch_first = True,
                 **kwargs) -> None:
        """
        Initialise the Multitask LSTM.

        :input_dims (base.List[int]): The dimensionality of the input.
        :shared_dim (int): The dimensionality of the shared layers.
        :hidden_dim (base.List[int]): The dimensionality of the hidden dimensions for each task.
        :embedding_dim (int): The dimensionality of the the produced embeddings.
        :no_classes: Number of classes for to predict on.
        :no_layers: The number of recurrent layers in the LSTM (1-3).
        :dropout: Value fo dropout
        """
        super(EmbeddingLSTMClassifier, self).__init__()
        self.name = "emb-mtl-lstm"
        self.batch_first = batch_first
        self.info = {'Input dim': ", ".join([str(it) for it in input_dims]), 'Embedding dim': embedding_dims,
                     'Shared dim': shared_dim, 'Hidden dim': ", ".join([str(it) for it in hidden_dims]),
                     'Output dim': ", ".join([str(it) for it in output_dims]),
                     '# layers': no_layers, 'Dropout': dropout, 'Model': self.name}

        assert len(input_dims) == len(output_dims)

        # Initialise the hidden dim
        self.all_parameters = nn.ParameterList()

        # Input layer (not shared) [Embedding]
        # hidden to hidden layer (shared) [Linear]
        # Hidden to hidden layer (not shared) [LSTM]
        # Output layer (not shared) [Linear}

        self.inputs = {}  # Define task inputs
        for task_id, input_dim in enumerate(input_dims):
            layer = nn.Embedding(input_dim, embedding_dims)
            self.inputs[task_id] = layer

            # Add parameters
            self.all_parameters.append(layer.weight)

        self.shared = nn.Linear(embedding_dims, shared_dim)
        self.all_parameters.append(self.shared.weight)
        self.all_parameters.append(self.shared.bias)

        self.lstm = {}
        for task_ix, _ in enumerate(input_dims):
            layer = nn.LSTM(shared_dim, hidden_dims[task_ix], batch_first = batch_first, num_layers = no_layers)
            self.lstm[task_ix] = layer

            # Add parameters
            self.all_parameters.append(layer.weight_ih_l0)
            self.all_parameters.append(layer.weight_hh_l0)
            self.all_parameters.append(layer.bias_ih_l0)
            self.all_parameters.append(layer.bias_hh_l0)

        self.outputs = {}
        for task_ix, _ in enumerate(input_dims):
            layer = nn.Linear(hidden_dims[task_ix], output_dims[task_ix])
            self.outputs[task_ix] = layer

            # Add parameters
            self.all_parameters.append(layer.weight)
            self.all_parameters.append(layer.bias)

        # Set the method for producing "probability" distribution.
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim = 1)

        # Ensure that the model is deterministic (the bias term is added)
        # print(self)
        # print(list(self.all_parameters))

    def forward(self, sequence, task_id, **kwargs) -> base.DataType:
        """
        Forward step in the classifier.

        :sequence: The sequence to pass through the network.
        :task_id: The task on which to perform forward pass.
        :return (base.DataType): The "probability" distribution for the classes.
        """
        if not self.batch_first:
            sequence = sequence.transpose(0, 1)

        res = self.inputs[task_id](sequence.long())
        res = self.shared(res)
        self.lstm[task_id].flatten_parameters()
        lstm_out, (lstm_hidden, _) = self.lstm[task_id](res)
        output = self.outputs[task_id](self.dropout(lstm_hidden))
        prob_dist = self.softmax(output)  # The probability distribution

        return prob_dist.squeeze(0)


class OnehotLSTMClassifier(nn.Module):
    """Multitask LSTM Classifier."""

    def __init__(self, input_dims: base.List[int], shared_dim: int, hidden_dims: base.List[int],
                 output_dims: base.List[int], no_layers: int = 1, dropout: float = 0.0, batch_first = True,
                 **kwargs) -> None:
        """
        Initialise the Multitask LSTM.

        :input_dims (base.List[int]): The dimensionality of the input.
        :shared_dim (int): The dimensionality of the shared layer.
        :hidden_dims (base.List[int]): The dimensionality of the hidden dimensions for each task.
        :output_dims (base.List[int]): Number of classes for to predict on.
        :no_layers (int, default = 1): The number of recurrent layers in the LSTM (1-3).
        :dropout (float, default = 0.0): Value of dropout layer.
        :batch_first (boo, default = True): If input tensors have the batch dimension in the first dimensino.
        """
        super(OnehotLSTMClassifier, self).__init__()
        self.name = "onehot-mtl-lstm"
        self.batch_first = batch_first
        self.info = {'Input dim': ", ".join([str(it) for it in input_dims]), 'Shared dim': shared_dim,
                     'Hidden dim': ", ".join([str(it) for it in hidden_dims]),
                     'Output dim': ", ".join([str(it) for it in output_dims]),
                     '# layers': no_layers, 'Dropout': dropout, 'Model': self.name}

        # Initialise the hidden dim
        self.all_parameters = nn.ParameterList()

        assert len(input_dims) == len(output_dims)

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

        self.shared = nn.Linear(shared_dim, shared_dim)
        self.all_parameters.append(self.shared.weight)
        self.all_parameters.append(self.shared.bias)

        # for i in range(len(hidden_dims) - 1):
        #     if i == 0:
        #         layer =
        #
        #     layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
        #     self.shared.append(layer)
        #
        #     # Add parameters
        #     self.all_parameters.append(layer.weight)
        #     self.all_parameters.append(layer.bias)

        self.lstm = {}
        for task_ix, _ in enumerate(input_dims):
            layer = nn.LSTM(shared_dim, hidden_dims[task_id], batch_first = batch_first, num_layers = no_layers)
            self.lstm[task_ix] = layer

            # Add parameters
            self.all_parameters.append(layer.weight_ih_l0)
            self.all_parameters.append(layer.weight_hh_l0)
            self.all_parameters.append(layer.bias_ih_l0)
            self.all_parameters.append(layer.bias_hh_l0)

        self.outputs = {}
        for task_id, _ in enumerate(input_dims):
            layer = nn.Linear(hidden_dims[task_id], output_dims[task_id])
            self.outputs[task_id] = layer

            # Add parameters
            self.all_parameters.append(layer.weight)
            self.all_parameters.append(layer.bias)

        # Set the method for producing "probability" distribution.
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim = 1)

        # Ensure that the model is deterministic (the bias term is added)
        print(self)
        print(list(self.all_parameters))

    def forward(self, sequence, task_id, **kwargs) -> base.DataType:
        """
        Forward step in the classifier.

        :sequence: The sequence to pass through the network.
        :task_id: The task on which to perform forward pass.
        :return (base.DataType): The "probability" distribution for the classes.
        """
        if not self.batch_first:
            sequence = sequence.transpose(0, 1)

        res = self.inputs[task_id](sequence.float())
        res = self.shared(res)
        self.lstm[task_id].flatten_parameters()
        lstm_out, (lstm_hidden, _) = self.lstm[task_id](res)
        output = self.outputs[task_id](self.dropout(lstm_hidden))
        prob_dist = self.softmax(output)  # The probability distribution

        return prob_dist.squeeze(0)


class OnehotMLPClassifier(nn.Module):
    """Onehot MLP MTL classifier."""

    def __init__(self, input_dims: base.List[int], shared_dim: int, hidden_dims: base.List[int],
                 output_dims: base.List[int], dropout: float = 0.0, batch_first = True, nonlinearity: str = 'tanh',
                 **kwargs) -> None:
        """
        Initialise the Multitask LSTM.

        :input_dims (base.List[int]): The dimensionality of the input.
        :shared_dim (int): The dimensionality of the shared layer.
        :hidden_dims (base.List[int]): The dimensionality of the hidden dimensions for each task.
        :output_dims (base.List[int]): Number of classes for to predict on.
        :dropout (float, default = 0.0): Value of dropout layer.
        :batch_first (boo, default = True): If input tensors have the batch dimension in the first dimensino.
        """
        super(OnehotMLPClassifier, self).__init__()
        self.batch_first = batch_first
        self.name = "mtl-onehot-mlp"
        self.info = {'Model': self.name,
                     'Input dim': ", ".join([str(it) for it in input_dims]), 'Hidden dim': hidden_dims,
                     'Output dim': ", ".join([str(it) for it in output_dims]), 'Shared dim': shared_dim,
                     'nonlinearity': nonlinearity, 'Dropout': dropout
                     }

        # Initialise the hidden dim
        self.all_parameters = nn.ParameterList()

        assert len(input_dims) == len(output_dims)

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

        self.shared = nn.Linear(shared_dim, shared_dim)
        self.all_parameters.append(self.shared.weight)
        self.all_parameters.append(self.shared.bias)

        self.hidden = {}
        for task_ix, _ in enumerate(input_dims):
            layer = nn.Linear(shared_dim, hidden_dims[task_ix])
            self.hidden[task_ix] = layer

            # Add parameters
            self.all_parameters.append(layer.weight)
            self.all_parameters.append(layer.bias)

        self.outputs = {}
        for task_id, _ in enumerate(input_dims):
            layer = nn.Linear(hidden_dims[task_ix], output_dims[task_id])
            self.outputs[task_id] = layer

            # Add parameters
            self.all_parameters.append(layer.weight)
            self.all_parameters.append(layer.bias)

        # Set the method for producing "probability" distribution.
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim = 1)
        self.nonlinearity = torch.tanh if nonlinearity == 'tanh' else torch.relu

        # Ensure that the model is deterministic (the bias term is added)
        # Uncomment to bughunt
        # print(self)
        # print(list(self.all_parameters))

    def forward(self, sequence, task_id, **kwargs) -> base.DataType:
        """
        Forward step in the classifier.

        :sequence: The sequence to pass through the network.
        :task_id: The task on which to perform forward pass.
        :return (base.DataType): The "probability" distribution for the classes.
        """
        if not self.batch_first:
            sequence = sequence.transpose(0, 1)

        res = self.inputs[task_id](sequence.float())
        res = self.shared(res)
        res = self.dropout(self.hidden[task_id](res))
        res = self.nonlinearity(res)

        # for layer in self.shared:
        #     res = self.dropout(layer(res))

        res = res.mean(-1)   # Reducing from (batch size, sequence length, 64) -> (batch size, sequence length)
        res = self.outputs[task_id](res)
        prob_dist = self.softmax(res)

        return prob_dist


class EmbeddingMLPClassifier(nn.Module):
    """Embedding MLP MTL classifier."""

    def __init__(self, input_dims: base.List[int], shared_dim: int, embedding_dims: int, hidden_dims: base.List[int],
                 output_dims: base.List[int], dropout: float = 0.0, batch_first: bool = True,
                 nonlinearity: str = 'tanh', **kwargs) -> None:
        """
        Initialise the Multitask LSTM.

        :input_dims (base.List[int]): The dimensionality of the input.
        :shared_dim (int): The dimensionality of the shared layer.
        :embedding_dims (base.List[int]): The dimensionality of the hidden dimensions for each task.
        :hidden_dims (base.List[int]): The dimensionality of the hidden dimensions for each task.
        :output_dims (base.List[int]): Number of classes for to predict on.
        :dropout (float, default = 0.0): Value of dropout layer.
        :batch_first (bool, default = True): If input tensors have the batch dimension in the first dimension.
        """
        super(EmbeddingMLPClassifier, self).__init__()
        self.name = "emb-mtl-mlp"
        self.batch_first = batch_first
        self.info = {'Input dim': ", ".join([str(it) for it in input_dims]), 'Shared dim': shared_dim,
                     'Embedding dim': embedding_dims,
                     'Output dim': ", ".join([str(it) for it in output_dims]),
                     'Dropout': dropout, 'Model': self.name, 'nonlinearity': nonlinearity}

        # Initialise the hidden dim
        self.all_parameters = nn.ParameterList()

        assert len(input_dims) == len(output_dims)

        # Input layer (not shared) [Embedding]
        # hidden to hidden layer (shared) [Linear]
        # Output layer (not shared) [Linear}

        self.inputs = {}  # Define task inputs
        for task_id, input_dim in enumerate(input_dims):
            layer = nn.Embedding(input_dim, embedding_dims)
            self.inputs[task_id] = layer

            # Add parameters
            self.all_parameters.append(layer.weight)

        self.shared = nn.Linear(embedding_dims, shared_dim)
        self.all_parameters.append(self.shared.weight)
        self.all_parameters.append(self.shared.bias)

        # self.shared = []
        # for i in range(len(hidden_dims)):
        #     if i == 0:
        #         layer = nn.Linear(embedding_dims, hidden_dims[i])
        #     else:
        #         layer = nn.Linear(hidden_dims[i - 1], hidden_dims[i])
        #     self.shared.append(layer)
        #
        #     # Add parameters
        #     self.all_parameters.append(layer.weight)
        #     self.all_parameters.append(layer.bias)

        self.hidden = {}
        for task_ix, _ in enumerate(input_dims):
            layer = nn.Linear(shared_dim, hidden_dims[task_ix])
            self.hidden[task_ix] = layer

            # Add parameters
            self.all_parameters.append(layer.weight)
            self.all_parameters.append(layer.bias)

        self.outputs = {}
        for task_id, _ in enumerate(input_dims):
            layer = nn.Linear(hidden_dims[-1], output_dims[task_id])
            self.outputs[task_id] = layer

            # Add parameters
            self.all_parameters.append(layer.weight)
            self.all_parameters.append(layer.bias)

        # Set the method for producing "probability" distribution.
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim = 1)
        self.nonlinearity = torch.tanh if nonlinearity == 'tanh' else torch.relu

        # Ensure that the model is deterministic (the bias term is added)
        # print(self)
        # print(list(self.all_parameters))

    def forward(self, sequence, task_id, **kwargs) -> base.DataType:
        """
        Forward step in the classifier.

        :sequence: The sequence to pass through the network.
        :task_id: The task on which to perform forward pass.
        :return (base.DataType): The "probability" distribution for the classes.
        """
        if not self.batch_first:
            sequence = sequence.transpose(0, 1)

        res = self.inputs[task_id](sequence)
        res = self.shared(res)
        res = self.dropout(self.hidden[task_id](res))
        res = self.nonlinearity(res)

        # for layer in self.shared:
        #     res = self.dropout(layer(res))

        res = res.mean(-1)
        res = self.outputs[task_id](res)
        prob_dist = self.softmax(res)

        return prob_dist
