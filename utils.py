"""Module for loading csv data, converting it into numpy array, and checking whether data is loaded properly."""
import pandas as pd
import numpy as np
from PIL import Image


def get_dataset():
    """
    Function for loading data from train.csv.

    Parameters
    ----------
        None

    Returns
    -------
        train_features : ndarray
            train data of MNIST. shape = (42000, 28, 28)
        train_label : ndarray
            train label of MNIST. shape = (42000,)
    """
    # Load train.csv by using pandas
    train_data = pd.read_csv("./data/train.csv")

    # Separate label from csv data
    train_features = train_data.copy()
    train_label = train_features.pop("label")

    # Change type of features and label into ndarray
    train_features = np.array(train_features)
    train_label = np.array(train_label)

    # Change the shape of train features from (, 784) to (, 28, 28)
    train_features = np.reshape(train_features, (-1, 28, 28))

    return train_features, train_label


# For debugging
if __name__ == "__main__":
    data, label = get_dataset()

    print(data.shape)
    print(label.shape)

    print(type(data))
    print(type(label))

    x = np.asarray(data[0], dtype=np.uint8)

    pil_image = Image.fromarray(x)
    pil_image.show()
