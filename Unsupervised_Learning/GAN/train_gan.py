import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def gen_noise(batch_size):
	'''
	This function will generate samples from a uniform distribution 
	'''
	noise = np.random.uniform(-1, 1, size=(batch_size, 128))
	return noise

def weight_initializer(shape, var_name):
	w = tf.compat.v1.get_variable(var_name, shape, tf.float32, initializer = tf.contrib.layers.xavier_initializer())  
	return w

def bias_initializer(shape, var_name):
	b = tf.compat.v1.get_variable(var_name, shape, tf.float32, initializer = tf.zeros_initializer)
	return b

def cost(logits, labels):
	cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
	return cost

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

def save_generated_images(generated_images, epoch):
	n_images = len(generated_images)
	for image_no in range(1, n_images+1):
		plt.imsave('./images/{}_{}'.format(epoch, image_no), (np.reshape(generated_images[image_no-1], (28, 28))*127.5 + 127.5), format='png', cmap='gray')

class Generator():
	def __init__(self):
		with tf.variable_scope('Generator'):
			self.G_w1, self.G_b1 = weight_initializer([128, 256], 'G_w1'), bias_initializer([256], 'G_b1')
			self.G_w2, self.G_b2 = weight_initializer([256, 784], 'G_w2'), bias_initializer([784], 'G_b2')

	def forward(self, Z):
		with tf.compat.v1.variable_scope('Generator_Forward'):
			# Input : batch_size X 128 ; Output : batch_size X 256
			G_h1 = tf.matmul(Z, self.G_w1) + self.G_b1
			G_a1 = tf.nn.leaky_relu(G_h1)
			# Input : batch_size X 256, Output : batch_size X 784
			G_h2 = tf.matmul(G_a1, self.G_w2) + self.G_b2
			G_a2 = tf.nn.tanh(G_h2)
			return G_a2

class Discriminator():
	def __init__(self):
		with tf.variable_scope('Discriminator', reuse=tf.compat.v1.AUTO_REUSE):
			self.D_w1, self.D_b1 = weight_initializer([784, 256], 'D_w1'), bias_initializer([256], 'D_b1')
			self.D_w2, self.D_b2 = weight_initializer([256, 1], 'D_w2'), bias_initializer([1], 'D_b2')

	def forward(self, X):
		with tf.compat.v1.variable_scope('Discriminator_Forward', reuse=tf.compat.v1.AUTO_REUSE):
			# Input : batch_size X 784, Output : batch_size X 256
			D_h1 = tf.matmul(X, self.D_w1) + self.D_b1
			D_a1 = tf.nn.leaky_relu(D_h1)
			# Input : batch_size X 256, Output : batch_size X 1
			D_h2 = tf.matmul(D_a1, self.D_w2) + self.D_b2
			D_a2 = tf.nn.sigmoid(D_h2)
			return D_h2

def train(epochs):
	# It is safe to clear the computation graph because there still could be variables present from the previous run  
	tf.compat.v1.reset_default_graph()

	# Creating summary object
	writer = tf.compat.v1.summary.FileWriter('./graph/')

	# X will store images from MNIST dataset
	X = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name='x')
	# Z will store samples generated from the normal distribution
	Z = tf.compat.v1.placeholder(tf.float32, shape=[None, 128], name='z')

	# Creating the Generator and Discriminator network
	generator, discriminator = Generator(), Discriminator()
	# Forwarding noise through the Generator Network
	G_out = generator.forward(Z)
	# Forwarding MNSIT dataset through the Discriminator network
	D_real_out = discriminator.forward(X)
	# Forwarding output from the Generator network through the Discriminator network
	D_fake_out = discriminator.forward(G_out)

	# Loss function and for the Discriminator
	with tf.name_scope('Discriminator_Loss'):
		D_real_loss, D_fake_loss = cost(D_real_out, tf.ones_like(D_real_out)*0.9), cost(D_fake_out, tf.zeros_like(D_fake_out))
		D_total_loss = D_real_loss + D_fake_loss
	# For showing discriminator loss on tensorboard
	tf.compat.v1.summary.scalar('Discriminator Loss', D_total_loss, family='loss')

	# Loss function for the Generator:
	with tf.name_scope('Generator_Loss'):
		G_loss = cost(D_fake_out, tf.ones_like(D_fake_out))
	# For showing generator loss on tensorboard
	tf.compat.v1.summary.scalar('Generator Loss', G_loss, family='loss')

	# Trainable parameters for Discriminator and Generator network
	D_train_params = [discriminator.D_w1, discriminator.D_b1, discriminator.D_w2, discriminator.D_b2]
	G_train_params = [generator.G_w1, generator.G_b1, generator.G_w2, generator.G_b2]

	# Optimize Discriminator and Generator network parameters
	with tf.name_scope('Discriminator_Optimization'):
		D_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.0003).minimize(D_total_loss, var_list = D_train_params)
	with tf.name_scope('Generator_Optimization'):
		G_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.0003).minimize(G_loss, var_list = G_train_params)

	# Loading and preparing MNIST dataset
	mnist = tf.keras.datasets.mnist
	(x_train, _), (x_test, _) = mnist.load_data()
	x = (np.append(x_train, x_test, axis=0) - 127.5)/ 127.5
	x = x.reshape(70000, 784).astype(np.float32)
	summary = tf.compat.v1.summary.merge_all()

	with tf.compat.v1.Session() as sess:

		# First we have to initialize all the variables present in our computational graph
		sess.run(tf.compat.v1.global_variables_initializer())

		# Object for model saving
		saver = tf.compat.v1.train.Saver()

		# Will train the entire system end to end for the specifed epochs
		for epoch in tqdm(range(1, epochs+1)):
			np.random.shuffle(x)
			# For one epoch with batch size as 128, we would need 546 iterations (546*128 = 69888)
			for iter in range(500):
				# Training the Discriminator alone
				real_images = x[128*iter:128*(iter+1)]
				# Getting next batch of samples from the normal (gaussian) distribution
				noise = gen_noise(128)
				_ = sess.run(D_optimizer, feed_dict = {X : real_images, Z : noise})

				# Training the Generator alone
				noise = gen_noise(128)
				# Training the Generator alone
				_ = sess.run(G_optimizer, feed_dict = {Z : noise})

			noise = gen_noise(x.shape[0])
			summ = sess.run(summary, feed_dict={X: x, Z: noise})
			writer.add_summary(summ, epoch)

			if epoch%50 == 0:
				# Will test our generator by producing 9 images
				noise = gen_noise(9)
				# Running the Generator network
				generated_images = sess.run(G_out, feed_dict = {Z : noise})
				# Will view the generated images using matplotlib
				#show_generated_images(generated_images)
				save_generated_images(generated_images, epoch)
			
			# Saving model after every 1 epoch
			if epoch%500 == 0:
				saver.save(sess, './model/gan_model', global_step=epoch)

		# For visualizing graph on tensorboard
		writer.add_graph(sess.graph)

if __name__ == '__main__':
	train(1000)