from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import csv
from PIL import Image
from PIL import ImageFile
import tensorflow as tf
import numpy as np
import tensorflow as tf


f  = open("labels.csv")
reader = csv.reader(f)
f2 = open("label_dict.csv")
reader2 = csv.reader(f2)
label_dict = {}
for line in reader2:
    label_dict[line[0]] = int(line[1])

train_data_input = []
eval_data_input = []
eval_labels_input = []
train_labels_input = []

reader.__next__()
counter = -1
for line in reader:
    counter += 1
    if counter % 100 == 0:
        print(counter)
    im = Image.open("./"+"trainModifiedGrainy"+"/"+line[0]+".jpg")
    toAppend = np.reshape(np.array(im).astype(float),[16384])
    if counter < 8000:
        train_data_input.append(toAppend)
        train_labels_input.append(label_dict[line[1]])
    else:
        eval_data_input.append(toAppend)
        eval_labels_input.append(label_dict[line[1]])

# train_data_input = tf.reshape(np.array(train_data_input), [-1,128,128,1])
# print(train_data_input.shape)


tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 128, 128, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 32 * 32 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.0, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=len(label_dict))

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = np.array(train_data_input) # mnist.train.images  # Returns np.array
  train_labels = np.array(train_labels_input)
  print(train_data.shape,train_labels.shape)
  eval_data = np.array(eval_data_input) # mnist.test.images  # Returns np.array
  eval_labels = np.array(eval_labels_input) # np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  dog_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="./modeldir")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)

  train_spec = tf.estimator.TrainSpec(
    input_fn = train_input_fn,
    max_steps = 20000
  )

  eval_spec = tf.estimator.EvalSpec(
    input_fn = eval_input_fn
  )

  tf.estimator.train_and_evaluate(
       dog_classifier,
       train_spec,
       eval_spec)

      # eval_results = dog_classifier.evaluate(input_fn=eval_input_fn)


if __name__ == "__main__":
  tf.app.run()
