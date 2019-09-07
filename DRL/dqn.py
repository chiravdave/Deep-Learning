#https://blog.floydhub.com/spinning-up-with-deep-reinforcement-learning/
#https://github.com/floodsung/DQN-Atari-Tensorflow/blob/master/BrainDQN_Nature.py
#
import tensorflow as tf

def weights_initialize(shape, var_name):
	w = tf.compat.v1.Variable(var_name, shape, tf.float32, initializer = tf.contrib.layers.xavier_initializer())
	return w

def bias_initialize(shape, var_name):
	b = tf.compat.v1.Variable(var_name, shape, tf.float32, initializer = tf.zeros_initializer)
	return b

class DQN:

	def __init__(self, network_name):
		self.w1, self.b1 = 