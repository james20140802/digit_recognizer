"""Module for loading csv data, converting it into numpy array, and checking whether data is loaded properly."""
import pandas as pd
import numpy as np
from PIL import Image


def get_dataset(path):
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


# For debugging
if __name__ == "__main__":
    image, label = get_dataset("./data/train.csv")

    print(image.shape)
    print(label.shape)

    print(type(image))
    print(type(label))

    x = np.asarray(image[0], dtype=np.uint8)
    x = np.reshape(x, (28, 28))

    pil_image = Image.fromarray(x)
    pil_image.show()
