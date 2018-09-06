import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import cv2
import sys
from preprocess import normalized_image
totaltime=0

#Reading file path containing images and labels
filepath = str(sys.argv[1]) + "/"
filename = 'labels.txt'

is_training = tf.placeholder(tf.int8,[],name = 'flag')
epsilon = 1e-3

def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training==1:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)

def conv2d(x, W, b, strides = 1):
    x = tf.nn.conv2d(x,W, strides=[1,strides,strides,1], padding='SAME')
    batch_normalized = batch_norm_wrapper(x,is_training)
    x_norm = tf.nn.bias_add(batch_normalized,b)
    return(tf.nn.relu(x_norm))

def conv2d_padding(x, W, b, strides = 1):
	x = tf.nn.conv2d(x,W, strides=[1,strides,strides,1], padding='VALID')
	x = tf.nn.bias_add(x,b)
	return(tf.nn.relu(x))

def maxpool2d(x, k=2):
	return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

def fclayer(x, W, b):
	fc1 = tf.add( tf.matmul(x,W), b,name = 'final')
	return fc1

def fclayer_relu(x, W, b):
    batch_normalized = batch_norm_wrapper(tf.matmul(x,W), is_training)
    x_norm = tf.nn.bias_add(batch_normalized, b)
    return tf.nn.relu(x_norm)

def init_w(shape):
	return tf.Variable(tf.truncated_normal(shape,mean=0.0, stddev = 0.1))
def init_b(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

x = tf.placeholder(tf.float32, shape=[None,196608],name = 'input')
y_ = tf.placeholder(tf.float32, shape=[None,2])

x_image = tf.reshape(x, [-1,256,256,3])
y_ = tf.reshape(y_, [-1,2])

w1 = init_w([3,3,3,8])
b1 = init_b([8])
conv1 = conv2d(x_image, w1, b1)

w2 = init_w([3,3,8,16])
b2 = init_b([16])
conv2 = conv2d(conv1, w2, b2)

maxp1 = maxpool2d(conv2)

w3 = init_w([3,3,16,32])
b3 = init_b([32])
conv3 = conv2d(maxp1, w3,b3)

w4 = init_w([3,3,32,64])
b4 = init_b([64])
conv4 = conv2d(conv3,w4,b4)

maxp2 = maxpool2d(conv4)

w5 = init_w([64,64,64,512])
b5 = init_b([512])
conv5 = conv2d_padding(maxp2,w5,b5)

conv5_flatten = tf.reshape(conv5,[-1,512])
w_fc1 = init_w([512,128])
b_fc1 = init_b([128])
fc1 = fclayer_relu(conv5_flatten,w_fc1, b_fc1)

#fc1 = tf.layers.dropout(fc1, rate=dropout, training = True)

w_fc2 = init_w([128,2])
b_fc2 = init_b([2])
fc2 = fclayer(fc1,w_fc2, b_fc2)

#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fc2))
loss = tf.reduce_sum(tf.square((fc2) - (y_)))

filenames, labels = normalized_image(filepath,filename)

input_image = []
sess = tf.InteractiveSession()
NumFiles = len(filenames)
#Converting filnames into tensor
tfilenames = ops.convert_to_tensor(filenames, dtype = dtypes.string)
tlabels = ops.convert_to_tensor(labels, dtype=dtypes.string)
#creating a queue which contains the list of files to read and the values of labels
filename_queue = tf.train.slice_input_producer([tfilenames, tlabels], num_epochs=10, shuffle=False, capacity = NumFiles)
#reading image files and decoding them
rawIm = tf.read_file("./" + filename_queue[0])
decodedIm = tf.image.decode_jpeg(rawIm)
lbl=[]
#extracting the labels queue
label_queue = filename_queue[1]
#Initializing global and local variable initializers
init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
#Creating an interactive session to run in python file
label_value = []
sess = tf.InteractiveSession()
sess.run(init_op)
no_epoch=0
loss_to_be_minimized = 0
label_counter = 0
train_step = tf.train.GradientDescentOptimizer(1e-8).minimize(loss)
saver = tf.train.Saver()
with sess.as_default():
    # start populating the filename queue
    # saver = tf.train.Saver()
    flag = 0  # epoch
    lbl_array = []
    img_array = []

    while (True):
        flag = flag + 1
        begtime = time.time()
        i = 0
        Train_Checker.append(loss_to_be_minimized)  # Previous loss function
        loss_to_be_minimized = 0
        for i in range(NumFiles):
            print i

            if flag <= 1:
                nm, image, lb = sess.run([filename_queue[0], decodedIm, label_queue])
                labels = np.reshape(labels, (-1, 2))
                lbl = labels[i]
                lbl_array.append(lbl)
                image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
                input_image = (sess.run(tf.reshape(image, [196608])))
                img_array.append(input_image)
                ip_img = input_image

            no_of_times_run = 0

            while (True):
                train_step.run(feed_dict={x: [img_array[i]], y_: [lbl_array[i]], is_training: 1})
                no_of_times_run = no_of_times_run + 1
                if no_of_times_run > 3:
                    break
            loss_to_be_minimized = loss_to_be_minimized + sess.run(loss,
                                                                   feed_dict={x: [img_array[i]], y_: [lbl_array[i]], is_training: 1})

        endtime = time.time()
        totaltime = totaltime + (endtime - begtime)
        no_epoch = no_epoch + 1

        print ("Epoch: " + str(flag) + "\t" + "Total Error: " + str(loss_to_be_minimized) + "\t" + "Tolerance: " + "\t" + "Time Taken: " + str(endtime - begtime))
        plt.ion()
        y = loss_to_be_minimized
        plt.xlabel("Epochs")
        plt.ylabel("Total_loss(L2 Loss)")
        plt.title("Loss Vs Epochs")
        plt.scatter(flag, y)
        plt.pause(0.05)
        if (no_epoch == 60 or loss_to_be_minimized < 800):
            break
    save_path = saver.save(sess, "./model")
    print ("Model saved in path: %s" % save_path)
    coord.request_stop()
    coord.join(threads)
    writer.close()
print("TotalTime Taken: " + str(totaltime))