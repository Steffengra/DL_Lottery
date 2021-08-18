
import tensorflow as tf
from abc import ABC

from DL_Lottery_imports.activation_functions import (
    activation_penalized_tanh
)


class ValueNetwork(tf.keras.Model, ABC):
    hidden_layers: list

    def __init__(
            self,
            name: str,
            num_hidden: list,
            activation: str,
            kernel_initializer: str
    ):
        super().__init__(name=name)
        # Activation----------------------------------------------------------------------------------------------------
        if activation == 'penalized_tanh':
            activation = activation_penalized_tanh
        # --------------------------------------------------------------------------------------------------------------

        # Layers--------------------------------------------------------------------------------------------------------
        self.hidden_layers = []
        for size in num_hidden:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    size,
                    activation=activation,
                    kernel_initializer=kernel_initializer,  # default: 'glorot_uniform'
                    # bias_initializer='zeros'  # default: 'zeros'
                ))

        self.output_layer = tf.keras.layers.Dense(1, dtype=tf.float32)
        # --------------------------------------------------------------------------------------------------------------

    @tf.function
    def call(
            self,
            inputs
    ) -> tf.Tensor:
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)

        return output

    def initialize_inputs(
            self,
            inputs
    ) -> None:
        """
        Ensure each method is traced once for saving
        """
        self(inputs)
        self.call(inputs)


class ValueNetworkBatchNorm(ValueNetwork, ABC):
    def __init__(
            self,
            name: str,
            num_hidden: list,
            activation: str,
            kernel_initializer: str
    ):
        super().__init__(
            name=name,
            num_hidden=num_hidden,
            activation=activation,
            kernel_initializer=kernel_initializer)

        self.len_hidden = len(self.hidden_layers)
        self.batch_norm_layers: list = []
        for _ in range(self.len_hidden):  # hidden - norm - hidden - norm - out
            self.batch_norm_layers.append(tf.keras.layers.BatchNormalization())

    @tf.function
    def call(
            self,
            inputs
    ) -> tf.Tensor:
        x = inputs
        for layer_id in range(self.len_hidden):
            x = self.hidden_layers[layer_id](x)
            x = self.batch_norm_layers[layer_id](x)

        output = self.output_layer(x)

        return output

    @tf.function
    def call_and_normalize_on_batch(
            self,
            inputs
    ) -> tf.Tensor:
        x = inputs
        for layer_id in range(self.len_hidden):
            x = self.hidden_layers[layer_id](x)
            x = self.batch_norm_layers[layer_id](x, training=True)

        output = self.output_layer(x)

        return output

    def initialize_inputs(
            self,
            inputs
    ) -> None:
        """
        Ensure each method is traced once for saving
        """
        self(inputs)
        self.call(inputs)
        self.call_and_normalize_on_batch(inputs)


class PolicyNetwork(tf.keras.Model, ABC):
    hidden_layers: list

    def __init__(
            self,
            name: str,
            num_hidden: list,
            num_actions: int,
            activation: str,
            kernel_initializer: str
    ) -> None:
        super().__init__(name=name)
        # Activation----------------------------------------------------------------------------------------------------
        if activation == 'penalized_tanh':
            activation = activation_penalized_tanh
        # --------------------------------------------------------------------------------------------------------------

        # Layers--------------------------------------------------------------------------------------------------------
        self.hidden_layers = []
        for size in num_hidden:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    size,
                    activation=activation,
                    kernel_initializer=kernel_initializer,  # default: 'glorot_uniform'
                    # bias_initializer='zeros'  # default: 'zeros'
                ))

        self.output_layer = tf.keras.layers.Dense(num_actions, activation='softmax', dtype=tf.float32)
        # --------------------------------------------------------------------------------------------------------------

    @tf.function
    def call(
            self,
            inputs,
    ) -> tf.Tensor:
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)

        return output

    def initialize_inputs(
            self,
            inputs
    ) -> None:
        """
        Ensure each method is traced once for saving
        """
        self(inputs)
        self.call(inputs)


class PolicyNetworkBatchNorm(PolicyNetwork, ABC):
    hidden_layers: list

    def __init__(
            self,
            name: str,
            num_hidden: list,
            num_actions: int,
            activation: str,
            kernel_initializer: str
    ):
        super().__init__(
            name=name,
            num_hidden=num_hidden,
            num_actions=num_actions,
            activation=activation,
            kernel_initializer=kernel_initializer)

        self.len_hidden = len(self.hidden_layers)
        self.batch_norm_layers: list = []
        for _ in range(self.len_hidden):  # hidden - norm - hidden - norm - out
            self.batch_norm_layers.append(tf.keras.layers.BatchNormalization())

    @tf.function
    def call(
            self,
            inputs
    ) -> tf.Tensor:
        x = inputs
        for layer_id in range(self.len_hidden):
            x = self.hidden_layers[layer_id](x)
            x = self.batch_norm_layers[layer_id](x)

        output = self.output_layer(x)

        return output

    @tf.function
    def call_and_normalize_on_batch(
            self,
            inputs
    ) -> tf.Tensor:
        x = inputs
        for layer_id in range(self.len_hidden):
            x = self.hidden_layers[layer_id](x)
            x = self.batch_norm_layers[layer_id](x, training=True)

        output = self.output_layer(x)

        return output

    def initialize_inputs(
            self,
            inputs
    ) -> None:
        """
        Ensure each method is traced once for saving
        """
        self(inputs)
        self.call(inputs)
        self.call_and_normalize_on_batch(inputs)
