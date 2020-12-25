import tensorflow as tf
from layers import weights_initialize, bias_initialize, conv2D, pool2D, fc_layer
from tensorflow import name_scope

class DQNModel:

	def __init__(self):
		with name_scope("dqn/weights_bias"):
			# Weights & Bias for first layer of convolution 
			self.w1, self.b1 = weights_initialize([3, 3, 4, 32], "w1"), bias_initialize([32], "b1")
			# Weights & Bias for second layer of convolution 
			self.w2, self.b2 = weights_initialize([3, 3, 32, 64], "w2"), bias_initialize([64], "b2")
			# Weights & Bias for the first FC layer
			self.w3, self.b3 = weights_initialize([6400, 512], "w3"), bias_initialize([512], "b3")
			# Weights & Bias for the final FC layer / Q-layer
			self.w4, self.b4 = weights_initialize([512, 2], "w4"), bias_initialize([2], "b4")

	def forward(self, inputs):
		with name_scope("dqn/layers"):
			# First layer of convolution
			conv1 = conv2D(inputs, self.w1, 2, "SAME", self.b1, "conv1", "act_1")
			# Second layer of convolution
			conv2 = conv2D(conv1, self.w2, 2, "SAME", self.b2, "conv2", "act_2")
			# First layer of pooling
			pool1 = pool2D(conv2, 2, 2, "SAME", "pool_2")
			# Flattening out the previous layer
			flatten = tf.reshape(pool1, [-1, 6400])
			# First FC layer
			fc1 = fc_layer(flatten, self.w3, self.b3, "tanh", "fc1", "act_3")
			# Final layer with Q-values
			q_values = fc_layer(fc1, self.w4, self.b4, "none", "q_values")
			return q_values