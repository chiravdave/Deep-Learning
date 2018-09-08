from preprocess import generateData
from sklearn.model_selection import train_test_split
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
    conv1 = conv_2Dlayer(input_layer, 64, 5, 1, 'same', tf.nn.relu)
    #pooling layer 1
    pool1 = pool_2Dlayer(conv1, 2, 2, 'valid')
    #Convolutional Layer 2
    conv2 = conv_2Dlayer(pool1, 64, 5, 1, 'same', tf.nn.relu)
    #pooling layer 2
    pool2 = pool_2Dlayer(conv2, 2, 2, 'valid')
    #Convolutional Layer 3
    conv3 = conv_2Dlayer(pool2, 128, 5, 1, 'same', tf.nn.relu)
    #pooling layer 3
    pool3 = pool_2Dlayer(conv3, 2, 2, 'valid')
    #Flatten all the feature maps for the dense layer
    flat = tf.layers.flatten(pool3)
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
    #Mode for dropout layer
    mode = tf.placeholder(shape=(), dtype=tf.bool)
    #Getting normalized features and labels
    features, labels = generateData(images_dir, image_width, image_height, channels)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
    #Getting predictions from the network
    predictions = my_network(input_layer, mode)
    #Calulating loss function
    loss = tf.losses.mean_squared_error(actual_labels, predictions)
    #Optimize the weigths and decrease the loss incurred
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    model = optimizer.minimize(loss)
    #Saver object to save our model once it is trained
    saver = tf.train.Saver()  
    #Storing Training & Testing Loss
    train_loss = []
    test_loss = []       
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())    #Initialize all the variables
        f = open("losses.txt", "a")                    #Saving loss information to a file
        for epoch in range(165):
            train_loss_run, _ = sess.run((loss, model), feed_dict={input_layer: x_train, actual_labels: y_train, mode : True})
            test_loss_run = sess.run(loss, feed_dict={input_layer: x_test, actual_labels: y_test, mode : False})
            f.write('Training Loss at {} iteration is: {}\n'.format((i+1), train_loss_run))
            f.write('Testing Loss at {} iteration is: {}\n\n'.format((i+1), test_loss_run))
            train_loss.append(train_loss_run)
            test_loss.append(test_loss_run)
            #Cross Validation
            x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
        save_path = saver.save(sess, "./model.ckpt")   #File name for our saved model
        f.close()
    fig = plt.figure()
    y = np.arange(1,166)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(y, train_loss, c='blue', label='Training')
    ax.scatter(y, test_loss, c='green', label='Testing')
    plt.title('Training vs Testing')                       #Plotting training vs testing accuracy
    plt.legend(loc=2)
    plt.show()

if __name__ == '__main__':
    train_network(sys.argv[1])
