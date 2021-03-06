import tensorflow as tf
import numpy as np

from util import gen_noise, show_generated_images

def test_gan():
	tf.reset_default_graph()
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph('./model/gan_model-1000.meta')
		saver.restore(sess, tf.train.latest_checkpoint('./model/'))
		graph = tf.get_default_graph()
		while True:
			Z = graph.get_tensor_by_name('z:0')
			gen_output = graph.get_tensor_by_name('Generator_Forward/Tanh:0')
			noise = gen_noise(9)
			generated_images = sess.run(gen_output, feed_dict = {Z: noise})
			show_generated_images(generated_images)
			choice = raw_input("Continue...?(y/n)")
			if choice != 'y':
				break


if __name__ == '__main__':
	test_gan()