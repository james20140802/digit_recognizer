"""Module for loading csv data, converting it into numpy array, and checking whether data is loaded properly."""
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image


def get_data(path):
    """
    Function for loading data from csv file

    Parameters
    ----------
        path : string
            path of the data(csv).

    Returns
    -------
        features : ndarray
            data of MNIST. shape = (, 28, 28, 1)
        label : ndarray
            label of MNIST. shape = (,)
    """
    # Load train.csv by using pandas
    data = pd.read_csv(path)

    # Separate label from csv data
    features = data.copy()
    label = features.pop("label")

    # Change type of features and label into ndarray
    features = np.array(features)
    label = np.array(label)

    # Change the shape of train features from (, 784) to (, 28, 28)
    features = np.reshape(features, (-1, 28, 28, 1))

    return features, label


def split_validation(features, label, validation_num):
    """
    Split data into two.(one for training, the other for validation)

    Parameters
    ----------
        features : ndarray
            data.
        label : ndarray
            data.
        validation_num : int
            number of validation data.

    Returns
    -------
        train_features : ndarray
        train_label : ndarray
        validation_features : ndarray
        validation_label : ndarray
    """
    train_features = features[validation_num:]
    train_label = label[validation_num:]

    validation_features = features[:validation_num]
    validation_label = label[:validation_num]

    return train_features, train_label, validation_features, validation_label


def preprocess(data):
    """
    Function for preprocessing image data.

    Parameters
    ----------
        data : ndarray
            image data.

    Returns
    -------
        preprocessed : ndarray
            preprocessed image data.
    """
    preprocessed = data / 225.0
    return preprocessed


def get_dataset(features, label, batch_size):
    """
    Change list(ndarray) into tensorflow Dataset.

    Parameters
    ----------
        features : ndarray
        label : ndarray
        batch_size : int

    Returns
    -------
        dataset : tf.data.Dataset
    """
    dataset = (
        tf.data.Dataset.from_tensor_slices((features, label))
        .shuffle(10000)
        .batch(batch_size)
    )

    return dataset


def get_train_dataset(path, validation_num, batch_size):
    """
    Get train dataset and validation data from path:

    Parameters
    ----------
        path : string
            path of the data(csv).
        validation_num : int
            number of validation data.
        batcb_size : int

    Returns
    -------
        train_dataset : tf.keras.Dataset
        validation_dataset : tf.keras.Dataset
    """
    features, label = get_data(path)

    features = preprocess(features)

    (
        train_features,
        train_label,
        validation_features,
        validation_label,
    ) = split_validation(features, label, validation_num)

    train_dataset = get_dataset(train_features, train_label, batch_size)
    validation_dataset = get_dataset(
        validation_features, validation_label, batch_size
    )

    return train_dataset, validation_dataset


# For debugging
if __name__ == "__main__":
    images, labels = get_data("./data/train.csv")

    print(images.shape)
    print(labels.shape)

    print(type(images))
    print(type(labels))

    x = np.asarray(images[0], dtype=np.uint8)
    x = np.reshape(x, (28, 28))

    pil_image = Image.fromarray(x)
    pil_image.show()
