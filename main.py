"""Module for main.py"""
import sys
import threading
import tensorflow as tf
import numpy as np
from PyQt5.QtWidgets import QApplication
from model import Model
from training import train_epoch
from validation import test_epoch
from validation import sample_test
from utils import get_train_dataset
from gui import Main


TRAIN_DATA_PATH = "./data/train.csv"
EPOCHS = 100
VALIDATION_NUM = 2000
LEARNING_RATE = 0.01
BATCH_SIZE = 16

model_ = Model()
loss_object_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimzer_ = tf.keras.optimizers.Adam(LEARNING_RATE)

metrics_loss = tf.keras.metrics.Mean()
metrics_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def train(
    train_data_loader,
    test_data_loader,
    model,
    loss_object,
    optimzer,
    epochs,
    train_metrics_loss,
    train_metrics_accuracy,
    validation_metrics_loss,
    validation_metrics_accuracy,
    gui=None,
):
    epoch_list = []
    train_loss = []
    train_accuracy = []
    validation_loss = []
    validation_accuracy = []
    sample_image_list = []
    sample_label_list = []
    sample_prediction_list = []

    tf.config.run_functions_eagerly(True)

    for epoch in range(1, epochs + 1):
        train_loss_new, train_accuracy_new = train_epoch(
            train_data_loader,
            model,
            loss_object,
            optimzer,
            train_metrics_loss,
            train_metrics_accuracy,
        )
        validation_loss_new, validation_accuracy_new = test_epoch(
            test_data_loader,
            model,
            loss_object,
            validation_metrics_loss,
            validation_metrics_accuracy,
        )

        sample_image, sample_label, sample_prediction = sample_test(
            test_data_loader, model
        )

        epoch_list.append(epoch)
        train_loss.append(train_loss_new)
        train_accuracy.append(train_accuracy_new)
        validation_loss.append(validation_loss_new)
        validation_accuracy.append(validation_accuracy_new)
        temp_sample_image = tf.squeeze(sample_image).numpy()
        temp_sample_image = temp_sample_image * 255
        sample_image_list.append(temp_sample_image.astype(np.uint8))
        sample_label_list.append(sample_label)
        sample_prediction_list.append(sample_prediction)
        if gui is not None:
            gui.update_metrics(
                epoch_list,
                train_accuracy,
                train_loss,
                validation_accuracy,
                validation_loss,
                sample_image_list,
                sample_label_list,
                sample_prediction_list,
            )
        else:
            pass


if __name__ == "__main__":
    train_dataset, test_dataset = get_train_dataset(
        TRAIN_DATA_PATH, VALIDATION_NUM, BATCH_SIZE
    )
    app = QApplication(sys.argv)
    ex = Main()
    t1 = threading.Thread(
        target=train,
        args=(
            train_dataset,
            test_dataset,
            model_,
            loss_object_,
            optimzer_,
            EPOCHS,
            metrics_loss,
            metrics_accuracy,
            metrics_loss,
            metrics_accuracy,
            ex,
        ),
    )
    t1.daemon = True
    t1.start()
    sys.exit(app.exec_())
