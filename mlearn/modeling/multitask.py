import torch.nn as nn
from mlearn import base


class EmbeddingLSTMClassifier(nn.Module):
    """Multitask LSTM Classifier."""

    def __init__(self, input_dims: base.List[int], embedding_dim: int, shared_dim: int, hidden_dims: base.List[int],
                 output_dims: base.List[int], no_layers: int = 1, dropout: float = 0.2, batch_first = True,
                 **kwargs) -> None:
        """
        Initialise the Multitask LSTM.

        :param input_dims (base.List[int]): The dimensionality of the input.
        :param shared_dim (int): The dimensionality of the shared layers.
        :param hidden_dim (base.List[int]): The dimensionality of the hidden dimensions for each task.
        :param embedding_dim (int): The dimensionality of the the produced embeddings.
        :param no_classes: Number of classes for to predict on.
        :param no_layers: The number of recurrent layers in the LSTM (1-3).
        :param dropout: Value fo dropout
        """
        super(EmbeddingLSTMClassifier, self).__init__()
        self.name = "emb-mtl-lstm"
        self.batch_first = batch_first

        # Initialise the hidden dim
        self.all_parameters = nn.ParameterList()

        assert len(input_dims) == len(hidden_dims) == len(output_dims)

        # Input layer (not shared) [Embedding]
        # hidden to hidden layer (shared) [Linear]
        # Hidden to hidden layer (not shared) [LSTM]
        # Output layer (not shared) [Linear}

        self.inputs = {}  # Define task inputs
        for task_id, input_dim in enumerate(input_dims):
            layer = nn.Embedding(input_dim, embedding_dim)
            self.inputs[task_id] = layer

            # Add parameters
            self.all_parameters.append(layer.weight)

        self.shared = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layer = nn.Linear(embedding_dim, hidden_dims[0])
            else:
                layer = nn.Linear(hidden_dims[i - 1], hidden_dims[i])
            self.shared.append(layer)

            # Add parameters
            self.all_parameters.append(layer.weight)
            self.all_parameters.append(layer.bias)

        self.lstm = {}
        for task_ix, hdim in enumerate(hidden_dims):  # TODO Double check this loop
            layer = nn.LSTM(hdim, hdim, batch_first = batch_first, num_layers = no_layers)
            self.lstm[task_ix] = layer

            # Add parameters
            self.all_parameters.append(layer.weight_ih_l0)
            self.all_parameters.append(layer.weight_hh_l0)
            self.all_parameters.append(layer.bias_ih_l0)
            self.all_parameters.append(layer.bias_hh_l0)

        self.outputs = {}
        for task_id, _ in enumerate(hidden_dims):
            layer = nn.Linear(hidden_dims[-1], output_dims[task_id])
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
        if not self.batch_first:
            sequence = sequence.transpose(0, 1)

        res = self.dropout(self.inputs[task_id](sequence.long()))

        for layer in self.shared:
            res = self.dropout(layer(res))

        lstm_out, (lstm_hidden, _) = self.lstm[task_id](res)

        output = self.outputs[task_id](lstm_hidden)

        prob_dist = self.softmax(output)  # The probability distribution

        return prob_dist.squeeze(0)


class OnehotLSTMClassifier(nn.Module):
    """Multitask LSTM Classifier."""

    def __init__(self, input_dims: base.List[int], shared_dim: int, hidden_dims: base.List[int],
                 output_dims: base.List[int], no_layers: int = 1, dropout: float = 0.2, batch_first = True,
                 **kwargs) -> None:
        """
        Initialise the Multitask LSTM.

        :param input_dims (base.List[int]): The dimensionality of the input.
        :param shared_dim (int): The dimensionality of the shared layer.
        :param hidden_dims (base.List[int]): The dimensionality of the hidden dimensions for each task.
        :param output_dims (base.List[int]): Number of classes for to predict on.
        :param no_layers (int, default = 1): The number of recurrent layers in the LSTM (1-3).
        :param dropout (float, default = 0.2): Value of dropout layer.
        :batch_first (boo, default = True): If input tensors have the batch dimension in the first dimensino.
        """
        super(OnehotLSTMClassifier, self).__init__()
        self.name = "onehot-mtl-lstm"
        self.batch_first = batch_first

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
            layer = nn.LSTM(hdim, hdim, batch_first = batch_first, num_layers = no_layers)  # Will go out of index.
            self.lstm[task_ix] = layer

            # Add parameters
            self.all_parameters.append(layer.weight_ih_l0)
            self.all_parameters.append(layer.weight_hh_l0)
            self.all_parameters.append(layer.bias_ih_l0)
            self.all_parameters.append(layer.bias_hh_l0)

        self.outputs = {}
        for task_id, _ in enumerate(hidden_dims):
            layer = nn.Linear(hidden_dims[-1], output_dims[task_id])
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
        if not self.batch_first:
            sequence = sequence.transpose(0, 1)

        res = self.dropout(self.inputs[task_id](sequence.float()))

        for layer in self.shared:
            res = self.dropout(layer(res))

        lstm_out, (lstm_hidden, _) = self.lstm[task_id](res)

        output = self.outputs[task_id](lstm_hidden)

        prob_dist = self.softmax(output)  # The probability distribution

        return prob_dist.squeeze(0)


class OnehotMLPClassifier(nn.Module):
    """Onehot MLP MTL classifier."""

    def __init__(self, input_dims: base.List[int], shared_dim: int, hidden_dims: base.List[int],
                 output_dims: base.List[int], dropout: float = 0.2, batch_first = True,
                 **kwargs) -> None:
        """
        Initialise the Multitask LSTM.

        :param input_dims (base.List[int]): The dimensionality of the input.
        :param shared_dim (int): The dimensionality of the shared layer.
        :param hidden_dims (base.List[int]): The dimensionality of the hidden dimensions for each task.
        :param output_dims (base.List[int]): Number of classes for to predict on.
        :param dropout (float, default = 0.2): Value of dropout layer.
        :batch_first (boo, default = True): If input tensors have the batch dimension in the first dimensino.
        """
        super(OnehotMLPClassifier, self).__init__()
        self.name = "onehot-mtl-mlp"
        self.batch_first = batch_first

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

        self.outputs = {}
        for task_id, _ in enumerate(hidden_dims):
            layer = nn.Linear(hidden_dims[-1], output_dims[task_id])
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

    def forward(self, sequence, task_id) -> base.DataType:
        """
        Forward step in the classifier.

        :param sequence: The sequence to pass through the network.
        :param task_id: The task on which to perform forward pass.
        :return (base.DataType): The "probability" distribution for the classes.
        """
        if self.batch_first:
            sequence = sequence.transpose(0, 1)

        res = self.dropout(self.inputs[task_id](sequence.float()))

        for layer in self.shared:
            res = self.dropout(layer(res))

        res = res.mean(0)
        res = self.outputs[task_id](res)
        prob_dist = self.softmax(res)

        return prob_dist
