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

def fc_layer(prev_layer, weights, bias, non_linearity, fc_layer="fc_layer", act_layer="act"):
	"""
	This function will help to initialize our fully connected layers.

	:param prev_layer: previous layer
	:param weights: weights for this layer
	:param bias: bias for this layer
	:param non_linearity: activation function
	:param fc_layer: fc layer name to be used in tensorflow graph
	:param act_layer: activation name to be used in tensorflow graph
	:rtype: fully connected layer
	"""

	out = tf.compat.v1.nn.bias_add(tf.matmul(prev_layer, weights), bias, name=fc_layer)
	if non_linearity == "relu":
		act = tf.compat.v1.nn.relu(out, name=act_layer)
		return act

	elif non_linearity == "leaky_relu":
		act = tf.compat.v1.nn.leaky_relu(out, name=act_layer)
		return act

	elif non_linearity == "tanh":
		act = tf.compat.v1.nn.tanh(out, name=act_layer)
		return act

	elif non_linearity == "none":
		return out
	
	else:
		raise Exception("Wrong function for non-linearity")