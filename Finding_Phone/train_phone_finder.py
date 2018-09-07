from preprocess import generateData
from sklearn.model_selection import train_test_split
import sys
import tensorflow as tf
import numpy as np

image_width = 490
image_height = 326
channels = 3

def conv_2Dlayer(inputs, filters, kernel_size, strides, pad, non_linearity):
    return tf.layers.conv2d(inputs, filters, kernel_size, strides=strides, padding=pad, activation=non_linearity)

def pool_2Dlayer(inputs, pooling_size, stride, pad):
    return tf.layers.max_pooling2d(inputs, pooling_size, strides=stride, padding=pad)

def dense_layer(inputs, units, non_linearity):
    return tf.layers.dense(inputs, units, activation=non_linearity)

def dropout_layer(inputs, drop_rate, train):
    return tf.layers.dropout(inputs, rate = drop_rate, training = train)

def my_network(input_layer, mode):
    #Convolutional Layer 1
    conv1 = conv_2Dlayer(input_layer, 32, 5, 1, 'same', tf.nn.relu)
    #pooling layer 1
    pool1 = pool_2Dlayer(conv1, 2, 2, 'valid')
    #Convolutional Layer 2
    conv2 = conv_2Dlayer(pool1, 64, 5, 1, 'same', tf.nn.relu)
    #pooling layer 2
    pool2 = pool_2Dlayer(conv2, 2, 2, 'valid')
    #Flatten all the feature maps for the dense layer
    flat = tf.layers.flatten(pool2)
    #Dense layer
    dense = dense_layer(flat, 1024, tf.nn.relu)
    #Dropout after dense layer
    dropout = dropout_layer(dense, 0.4, mode)
    #center value dense layer
    center_values = dense_layer(dropout, 2, None)
    return center_values

def train_network(images_dir):
    global image_width, image_height, channels
    input_layer = tf.placeholder(shape=(None, image_height, image_width, channels), dtype = tf.float32)
    actual_labels = tf.placeholder(shape=(None, 2), dtype = tf.float32)
    mode = tf.placeholder(shape=(), dtype=tf.bool)
    #Getting normalized features and labels
    features, labels = generateData(images_dir, image_width, image_height, channels)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)
    #Getting predictions from the network
    predictions = my_network(input_layer, mode)
    #Calulating loss function
    loss = tf.losses.mean_squared_error(actual_labels, predictions)
    #Optimize the weigths and decrease the loss incurred
    optimizer = tf.train.AdamOptimizer()
    model = optimizer.minimize(loss)
    #Saver object to save our model once it is trained
    saver = tf.train.Saver()        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())    #Initialize all the variables
        loss_run, _ = sess.run((loss, model), feed_dict={input_layer: x_train[0:1], actual_labels: y_train[0:1], mode : True})
        print(loss)
        #save_path = saver.save(sess, "./model.ckpt")   #File name for our saved model

if __name__ == '__main__':
    train_network(sys.argv[1])
