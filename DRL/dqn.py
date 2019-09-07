def weights_initialize(shape):
	weights = tf.Variable(tf.truncated_normal(shape))
	return weights

def bias_initialize(neurons):
	bias = tf.Variable(tf.zeros(neurons))
	return bias

def fc_layer(prev_layer, input_size, output_size):
	weights = weights_initialize([input_size, output_size])
	bias = bias_initialize(output_size)
	return tf.nn.relu(tf.matmul(prev_layer, weights) + bias)

class DQN:

	def __init__(self):
		pass