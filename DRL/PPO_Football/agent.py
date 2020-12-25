import tensorflow as tf

from layers import weights_initialize, bias_initialize, conv2D, pool2D, fc_layer


class PPOAgent(tf.Keras.Model):
	def __init__(self, inputs, actions: List[str]):
		super(PPOAgent, self).__init__()
		self.actions = self.actions
		self.policy, self.value = self._create_model(inputs)

	def _create_model(self, inputs):
		n_actions = len(self.actions)
		with tf.compat.v1.name_scope("actor_critic"):
			# Filters for first Conv layer 
			w1, b1 = weights_initialize([3, 3, 16, 32], "w1"), bias_initialize([32], "b1")
			# Filters for second Conv layer
			w2, b2 = weights_initialize([3, 3, 32, 64], "w2"), bias_initialize([64], "b2")
			# Filters for third Conv layer
			w3, b3 = weights_initialize([3, 3, 64, 128], "w3"), bias_initialize([128], "b3")
			# Filters for fourth Conv layer
			w4, b4 = weights_initialize([3, 3, 128, 256], "w4"), bias_initialize([256], "b4")
			# Weights & Bias for the first FC layer
			w5, b5 = weights_initialize([6400, 256], "w5"), bias_initialize([256], "b5")
			# Weights & Bias for the raw policy layer
			w6, b6 = weights_initialize([256, n_actions], "w6"), bias_initialize([n_actions], "b6")
			# Weights & Bias for the value layer
			w7, b7 = weights_initialize([256, 1], "w7"), bias_initialize([1], "b7")
			
			# First Conv layer
			conv1 = conv2D(inputs, w1, 1, "SAME", b1, "conv1", "act_1")
			# Second Conv layer
			conv2 = conv2D(conv1, w2, 2, "SAME", b2, "conv2", "act_2")
			# First Pooling layer
			pool1 = pool2D(conv2, 2, 2, "SAME", "pool_2")
			# Third Conv layer
			conv3 = conv2D(pool1, w3, 1, "SAME", b3, "conv3", "act_3")
			# Forth Conv layer
			conv4 = conv2D(conv3, w4, 2, "SAME", b4, "conv4", "act_4")
			# Flattening out the previous layer
			flatten = tf.reshape(conv4, [-1, 6400])
			# First FC layer
			fc1 = fc_layer(flatten, w5, b5, "relu", "fc1", "act_5")
			# Policy layer with action values
			raw_policy = fc_layer(fc1, w6, b6, "relu", "raw_policy")
			# Value layer with value of a state
			value = fc_layer(fc1, w7, b7, "tanh", "value_layer")
			
			return tf.nn.softmax(raw_policy, name="final_policy"), value

	def call(self, inputs):
		pass
