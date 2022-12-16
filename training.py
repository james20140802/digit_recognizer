"""Module for training model."""
import tensorflow as tf


def train_epoch(
    data_loader,
    model,
    loss_object,
    optimizer,
    train_metrics_loss,
    train_metrics_accuracy,
):
    """
    Function for training model in one epoch.

    Parameters
    ----------
        data_loader : tf.data.Dataset
            dataset for training.
        model : tf.keras.Model
            model that will be trained.
        loss_object : tf.keras.losses
        optimizer : tf.keras.optimizers
        train_metrics_loss : tf.keras.metrics
        train_metrics_accuracy : tf.keras.metrics

    Returns
    -------
        train_loss : scalar tensor
            loss during train.
        train_accuracy : scalar tensor
            accuracy during train.
    """
    # Reset the metrics at the start of epoch.
    train_metrics_loss.reset_state()
    train_metrics_accuracy.reset_state()

    # Train model.
    for images, labels in data_loader:
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradient(zip(gradients, model.trainable_variables))

        # Update state of metircs.
        train_metrics_loss(loss)
        train_metrics_accuracy(labels, predictions)

    train_loss = train_metrics_loss.result()
    train_accuracy = train_metrics_accuracy.result()

    return train_loss, train_accuracy
