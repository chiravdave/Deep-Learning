import tensorflow as tf
import numpy as np
from preprocess import normalizeImage
import cv2
import sys

def test_network(image_path):
    #creating a session
    sess = tf.Session()
    #restoring my trained model
    saver = tf.train.import_meta_graph("./model.ckpt.meta")
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    #getting my variables from the trained model
    x = graph.get_tensor_by_name("Placeholder:0")
    prediction = graph.get_tensor_by_name("dense_1/BiasAdd:0")
    mode = graph.get_tensor_by_name("Placeholder_2:0")
    bgr_image = cv2.imread(image_path)
    b, g, r = cv2.split(bgr_image)      #get b,g,r values of the image
    rgb_image = cv2.merge([r,g,b])      #convert into r,g,b format
    normalized_image = normalizeImage(rgb_image)
    reshaped_normalized_image = np.reshape(normalized_image, (-1, 326, 490, 3))
    #storing my results
    feed_dict = {x:reshaped_normalized_image, mode:False}
    center_values = sess.run(prediction, feed_dict)
    #getting my center coordinates
    cx = center_values[0][0]
    cy = center_values[0][1]
    #displaying final results
    print (cx, cy)

if __name__ == '__main__':
    test_network(sys.argv[1])
