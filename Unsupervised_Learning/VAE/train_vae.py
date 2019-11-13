import tensorflow as tf
import numpy as np
import time

from tqdm import tqdm
from layers import *
from util import gen_noise, show_generated_images, save_generated_images

class VAE():

	def __init__(self):
		with tf.compat.v1.variable_scope('Encoder'):
			# Weights and Bias for hidden layer 1
			self.h1_w, self.h1_b = weights_initialize([784, 256], 'h1_w'), bias_initialize([256], 'h1_b')
			# Weights and Bias for the mean vector
			self.mean_w, self.mean_b = weights_initialize([256, 10], 'mean_w'), bias_initialize([10], 'mean_b')
			# Weights and Bias for the standard deviation vector
			self.std_w, self.std_b = weights_initialize([256, 10], 'std_w'), bias_initialize([10], 'std_b')

		with tf.compat.v1.variable_scope('Decoder'):
			# Weights and Bias for hidden layer 1
			self.h2_w, self.h2_b = weights_initialize([10, 256], 'h1_w'), bias_initialize([256], 'h1_b')
			# Weights and Bias for the output layer
			self.out_w, self.out_b = weights_initialize([256, 784], 'out_w'), bias_initialize([784], 'out_b')

	def encoder(self, X):
		with tf.compat.v1.variable_scope('Encoder'):
			# Input : batch_size X 784; Output : batch_size X 256
			enc_h1 = fc_layer(X, self.h1_w, self.h1_b, "leaky_relu", act_layer="h1_layer")
			# Input : batch_size X 256, Output : batch_size X 10
			mean = fc_layer(enc_h1, self.mean_w, self.mean_b, "tanh", act_layer="mean")
			# Input : batch_size X 256, Output : batch_size X 10
			std = 0.5 * fc_layer(enc_h1, self.std_w, self.std_b, "tanh", act_layer="std")

			# Reparameterization trick
			epsilon = tf.random.normal(tf.shape(std), name="epsilon")
			z = mean + tf.multiply(epsilon, tf.exp(std))
			return mean, std, z

	def decoder(self, Z):
		with tf.compat.v1.variable_scope('Decoder'):
			# Input : batch_size X 10; Output : batch_size X 256
			dec_h1 = fc_layer(Z, self.h2_w, self.h2_b, "leaky_relu", act_layer="h1_layer")
			# Input : batch_size X 256, Output : batch_size X 784
			out = fc_layer(dec_h1, self.out_w, self.out_b, "leaky_relu", act_layer="out")
			return out

def train(epochs):
	# It is safe to clear the computation graph because there still could be variables present from the previous run  
	tf.compat.v1.reset_default_graph()

	# Creating summary object
	writer = tf.compat.v1.summary.FileWriter('./graph/')

	# X will store images from MNIST dataset
	X = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name='x')

	# Creating the VAE network
	variational_autoencoder = VAE()
	# Forwarding MNSIT dataset through the Encoder
	mean, std, z = variational_autoencoder.encoder(X)
	# Forwarding latent values from the encoder to the decoder
	out = variational_autoencoder.decoder(z)

	# Loss function for VAE
	with tf.name_scope('Total_Loss'):
		# Reconstruction loss
		with tf.name_scope('Reconstruction_Loss'):
			reconstruction_loss = tf.reduce_mean(tf.compat.v1.squared_difference(out, X))
		# KL Divergence loss
		with tf.name_scope('KL_Divergence_Loss'):
			kl_divergence_loss = -0.5 * tf.reduce_mean(1 + (2 * std) - tf.square(mean) - tf.exp(2 * std), 1)
		# For showing reconstruction, KL Divergence and total loss on tensorboard
		total_loss = tf.reduce_mean(reconstruction_loss + kl_divergence_loss)

	tf.compat.v1.summary.scalar('Reconstruction Loss', reconstruction_loss, family='loss')
	# tf.compat.v1.summary.scalar('KL Divergence Loss', kl_divergence_loss, family='loss')
	tf.compat.v1.summary.scalar('Total Loss', total_loss, family='loss')

	# Optimize Discriminator and Generator network parameters
	with tf.name_scope('Optimization'):
		optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.0003)
		minimize_loss = optimizer.minimize(total_loss)

	# Loading and preparing MNIST dataset
	mnist = tf.keras.datasets.mnist
	(x_train, _), (x_test, _) = mnist.load_data()
	x = np.append(x_train, x_test, axis=0)
	x = x.reshape(70000, 784).astype(np.float32)
	summary = tf.compat.v1.summary.merge_all()

	with tf.compat.v1.Session() as sess:

		# First we have to initialize all the variables present in our computational graph
		sess.run(tf.compat.v1.global_variables_initializer())
		# For visualizing graph on tensorboard
		writer.add_graph(sess.graph)
		# Object for model saving
		saver = tf.compat.v1.train.Saver()

		# Will train the entire system end to end for the specifed epochs
		for epoch in tqdm(range(1, epochs+1)):
			np.random.shuffle(x)
			# For one epoch with batch size as 128, we would need 546 iterations (546*128 = 69888)
			for iteration in range(500):
				# Getting next batch of images
				next_batch = x[128*iteration:128*(iteration+1)]
				# Training the VAE network
				sess.run(minimize_loss, feed_dict = {X : next_batch})

			summ = sess.run(summary, feed_dict={X: next_batch})
			writer.add_summary(summ, epoch)

			if epoch%300 == 0:
				# Will test our VAE by producing 9 images
				noise = gen_noise(9)
				# Running the VAE network
				generated_images = sess.run(out, feed_dict = {z : noise})
				# Will view the generated images using matplotlib
				# show_generated_images(generated_images)
				save_generated_images(generated_images, epoch)
			
			# Saving model after every 1 epoch
			if epoch%2500 == 0:
				saver.save(sess, './model/vae', global_step=epoch)

if __name__ == '__main__':
	train(5000)