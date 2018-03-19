import tensorflow as tf
import numpy as np
from preprocess import normalized_testimages
import cv2
import sys

#creating a session
sess = tf.Session()

#restoring my trained model
saver = tf.train.import_meta_graph("model.meta")
saver.restore(sess,tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()

#getting my variables from the trained model
x = graph.get_tensor_by_name("input:0")
layer = graph.get_tensor_by_name("final:0")
is_training = graph.get_tensor_by_name("flag:0")

#reading the location of the test file
testPath = str(sys.argv[1])

img = cv2.imread(testPath)
img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
img_resized = normalized_testimages(img_resized)
#storing my results
prediction = np.zeros(1)

image_linear = np.reshape(img_resized, [-1, 196608])
feed_dict = {x:image_linear, is_training:0}
prediction = (sess.run(layer, feed_dict))

#getting my center coordinates
ix = prediction[0][0]
iy = prediction[0][1]

#rounding the coordinate values till 4 decimal places
normalized_x = round(ix/256.0,4)
normalized_y = round(iy/256.0,4)

#displaying final results
print normalized_x, normalized_y