import tensorflow as tf

xavier_initializer = tf.keras.initializers.GlorotNormal()

def weights_initialize(shape, var_name="w"):
	"""
	This function will help to initialize our weights.

	:param shape: shape/dimension of the weight matrix
	:param var_name: name to be used in tensorflow graph for the weights
	:rtype: weight matrix
	"""

	return tf.Variable(xavier_initializer(shape=shape, dtype=tf.float32), name=var_name, shape=shape, dtype=tf.float32)

def bias_initialize(shape, var_name="b"):
	"""
	This function will help to initialize our bias.

	:param shape: shape/dimension of the bias
	:param var_name: name to be used in tensorflow graph for the bias 
	:rtype: bias
	"""

	return tf.Variable(tf.zeros(shape=shape, dtype=tf.float32), name=var_name, shape=shape, dtype=tf.float32)

def conv2D(prev_layer, kernel, stride, pad, bias, conv_layer="conv_layer", act_layer="act"):
	"""
	This function will help to initialize our convolutional layers.

	:param prev_layer: previous layer
	:param kernel: kernels for this layer
	:param stride: stride for this layer
	:param pad: padding for this layer
	:param bias: bias for this layer
	:param conv_layer: conv layer name to be used in tensorflow graph
	:param act_layer: activation name to be used in tensorflow graph
	:rtype: convolutional layer
	"""

	conv = tf.compat.v1.nn.bias_add(tf.compat.v1.nn.conv2d(prev_layer, filter=kernel, strides=stride, padding=pad), bias, name=conv_layer)
	return tf.compat.v1.nn.relu(conv, name=act_layer)

def pool2D(prev_layer, k_size, stride, pad, pool_layer="pool_layer"):
	"""
	This function will help to initialize our pooling layers.

	:param prev_layer: previous layer
	:param k_size: kernel size for this layer
	:param stride: stride for this layer
	:param pad: padding for this layer
	:param pool_layer: pool layer name to be used in tensorflow graph
	:rtype: pooling layer
	"""

	return tf.compat.v1.nn.max_pool2d(prev_layer, k_size, stride, pad, name=pool_layer) 

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
		return tf.compat.v1.nn.relu(out, name=act_layer)

	elif non_linearity == "sigmoid":
		return tf.compat.v1.nn.sigmoid(out, name=act_layer)

	elif non_linearity == "tanh":
		return tf.math.tanh(out, name=act_layer)
	
	elif non_linearity == "none":
		return out
	
	else:
		raise Exception("Wrong function for non-linearity")