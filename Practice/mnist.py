from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


# Our application logic will be added here
def mnist_model(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])  # Reshape input into [bacth_size, width, height, n_channels]

    # Convolutional Layer 1
    conv_1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=5, padding="same", activation=tf.nn.relu)

    # Pooling Layer 1
    pool_1 = tf.layers.max_pooling2d(inputs=conv_1, pool_size=2, strides=2)

    # Convolutional Layer 2
    conv_2 = tf.layers.conv2d(inputs=pool_1, filters=64, kernel_size=5, padding="same", activation=tf.nn.relu)

    # Pooling Layer 2
    pool_2 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=2, strides=2)

    # Dense Layer 1
    flatten = tf.reshape(pool_2, [-1, 3136])  # 7X7X64 = 3136
    dense_1 = tf.layers.dense(inputs=flatten, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense_1, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Final Dense Layer
    logits = tf.layers.dense(inputs=dropout, units=10)
    prediction = {"classes": tf.argmax(input=logits, axis=1), "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

    # Testing
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=prediction)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Training
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Evaluating
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=prediction["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    # Creating the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=mnist_model,
                                              model_dir="/tmp/mnist_convnet_model")  # model_dir will be used to store checkpoints
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
