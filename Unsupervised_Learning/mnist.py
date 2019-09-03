import tensorflow as tf
import numpy as np

def conv_2Dlayers(inputs, filters, kernel_size, stride, pad, non_linearity):
  return tf.layers.conv2d(inputs, filters, kernel_size, strides = stride, padding = pad, activation = non_linearity)

def pool_2Dlayer(inputs, pooling_size, stride, pad):
  return tf.layers.max_pooling2d(inputs, pooling_size, strides = stride, padding = pad)

def dense_layer(inputs, units, non_linearity):
  return tf.layers.dense(inputs, units, activation = non_linearity)

def dropout_layer(inputs, drop_rate, train):
  return tf.layers.dropout(inputs, rate = drop_rate, training = train)

def my_network(features, mode):
  input_layer =  tf.reshape(features, [-1,28,28,1])
  #Convolutional Layer 1
  conv1 = conv_2Dlayers(input_layer, 32, 5, 1, 'same', tf.nn.relu)
  #pooling layer 1
  pool1 = pool_2Dlayer(conv1, 2, 2, 'valid')
  #Convolutional Layer 2
  conv2 = conv_2Dlayers(pool1, 64, 5, 1, 'same', tf.nn.relu)
  #pooling layer 2
  pool2 = pool_2Dlayer(conv2, 2, 2, 'valid')
  #Flatten all the feature maps for the dense layer
  flat = tf.reshape(pool2, [-1, 7*7*64])
  #Dense layer
  dense = dense_layer(flat, 1024, tf.nn.relu)
  #Dropout after dense layer
  dropout = dropout_layer(dense, 0.4, mode)
  #Logits dense layer
  logits = dense_layer(dropout, 10, None)
  return logits

def train_test():
  # Load training and testing data
  mnist_data = tf.contrib.learn.datasets.load_dataset('mnist')
  features = tf.placeholder(shape=(None, 784), dtype=tf.float32)
  labels = tf.placeholder(shape=(None), dtype=tf.int64)
  mode = tf.placeholder(shape=(), dtype=tf.bool)
  logits = my_network(features, mode)
 
  #Finding the prediction label
  label_pred = tf.argmax(logits, axis=1)

  #Finding probabilities of the predicted labels
  label_prob = tf.nn.softmax(logits)

  #Calulate loss function
  one_hot_label = tf.one_hot(labels, depth = 10, dtype=tf.int64) # Depth = No.of Classes
  loss = tf.losses.softmax_cross_entropy(one_hot_label, label_prob)

  #Optimize the weigths and decrease the loss incurred
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
  optimizer = tf.train.AdamOptimizer()
  model = optimizer.minimize(loss)

  #Evaluate the accuracy of the predictions by the model
  accu_bool = tf.equal(labels, label_pred)
  accuracy = tf.reduce_mean(tf.cast(accu_bool, tf.float32))
  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(30001):
      train_data = mnist_data.train.next_batch(120) # Creating batches for training
      loss_run, _ = sess.run((loss, model), feed_dict={features: train_data[0], labels: train_data[1], mode : True})
      if((i+1)%500 == 0):
        print('Loss at {} epoch is: {}'.format((i+1)/500,loss_run))
    for i in range(1):
      test_data = mnist_data.test.next_batch(10000) # Creating batches for testing
      accurate = sess.run(accuracy, feed_dict={features: test_data[0], labels: test_data[1], mode : False})
      print('Accuracy at {} iteration is: {}'.format((i+1),accurate))

if __name__ == '__main__':
  train_test()
