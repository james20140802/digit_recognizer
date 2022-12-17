import tensorflow as tf


@tf.function
def test_epoch(
    data_loader,
    model,
    loss_object,
    test_metrics_loss,
    test_metrics_accuracy,
):
    """
    Function for testing(validation) model in one epoch.

    Parameters
    ----------
        data_loader : tf.data.Dataset
            dataset for testing.
        model : tf.keras.Model
            model that will be tested.
        loss_object : tf.keras.losses
        test_metrics_loss : tf.keras.metrics
        test_metrics_accuracy : tf.keras.metrics

    Returns
    -------
        test_loss : scalar tensor
            loss during test.
        test_accuracy : scalar tensor
            accuracy during test.
    """
    # Reset the metrics at the start of epoch.
    test_metrics_loss.reset_state()
    test_metrics_accuracy.reset_state()

    # Train model.
    for images, labels in data_loader:
        predictions = model(images, training=False)
        loss = loss_object(labels, predictions)

        # Update state of metircs.
        test_metrics_loss(loss)
        test_metrics_accuracy(labels, predictions)

    test_loss = test_metrics_loss.result()
    test_accuracy = test_metrics_accuracy.result()

    test_loss = test_loss.numpy()
    test_accuracy = test_accuracy.numpy()

    return test_loss, test_accuracy


def sample_test(data_loader, model):
    dataset_temp = data_loader.shuffle(
        1000, reshuffle_each_iteration=True
    ).unbatch()
    dataset_temp = list(dataset_temp.as_numpy_iterator())

    sample_data, sample_label = dataset_temp[0]
    sample_prediction = model(sample_data[tf.newaxis, ...], training=False)

    sample_prediction = tf.math.argmax(
        tf.squeeze(tf.nn.softmax(sample_prediction))
    ).numpy()

    return sample_data, sample_label, sample_prediction
