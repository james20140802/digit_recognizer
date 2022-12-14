"""Module for building deep learning model."""
import tensorflow as tf


class Model(tf.keras.Model):
    """
    A class to build CNN model.

    Attributes
    ----------
        conv1 : tf.keras.layers.Conv2D
            first cnn layer.
        flatten : tf.keras.layers.Flatten
            flatten the result of cnn layers.
        dense1 : tf.keras.layers.Dense
            first dense layer.
        dense2 : tf.keras.layers.Dense
            second and last dense layer.

    Methods
    -------
        call(x):
            return the output of model.
    """

    def __init__(self):
        """
        Initialize model.

        Parameters
        ----------
            None
        """
        super(Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        """
        Call the model on new inputs and return the output as tensors.

        Parameters
        ----------
            inputs : tensors
                input tensor, or dict/list/tuple of input tensors.
            training : boolean
                indicating whether to run the Network in training mode or inference mode.
            mask : list of boolean or none
                a mask or list of masks.

        Returns
        ------
            output : tensor or list of tensors
                output of the model.
        """
        x = self.conv1(inputs)
        x = self.dense1(x)
        output = self.dense2(x)
        return output
