import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def gen_noise(batch_size):
	'''
	This function will generate samples from a uniform distribution 
	'''
	noise = np.random.uniform(-1, 1, size=(batch_size, 128))
	return noise

def show_generated_images(generated_images):
		'''
		This method will show 3 images in a single row, so n_images should be divisible by 3
		'''

		n_images = len(generated_images)
		rows = int(n_images/3)
		for image_no in range(1, n_images+1):
		  plt.subplot(rows, 3, image_no)
		  plt.imshow((np.reshape(generated_images[image_no-1], (28, 28))*127.5 + 127.5), cmap='gray')
		plt.show()

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