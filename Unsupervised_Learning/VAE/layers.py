import tensorflow as tf

xavier_initializer = tf.glorot_normal_initializer()

def weights_initialize(shape, var_name):
	"""
	This function will help to initialize our weights.

	:param shape: shape/dimension of the weight matrix
	:param var_name: name to be used in tensorflow graph for the weights
	:rtype: weight matrix
	"""

	w = tf.compat.v1.Variable(xavier_initializer(shape=shape, dtype=tf.float32), name=var_name, shape=shape, dtype=tf.float32)
	return w

def bias_initialize(shape, var_name):
	"""
	This function will help to initialize our bias.

	:param shape: shape/dimension of the bias
	:param var_name: name to be used in tensorflow graph for the bias 
	:rtype: bias
	"""

	b = tf.compat.v1.Variable(tf.zeros(shape=shape, dtype=tf.float32), name=var_name, shape=shape, dtype=tf.float32)
	return b