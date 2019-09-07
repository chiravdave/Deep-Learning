import tensorflow as tf
from numpy.random import uniform, randint
import cv2
from memory import ReplayBuffer
from preprocessing import preprocess_frame

def weights_initialize(shape, var_name):
	w = tf.compat.v1.Variable(var_name, shape, tf.float32, initializer = tf.contrib.layers.xavier_initializer())
	return w

def bias_initialize(shape, var_name):
	b = tf.compat.v1.Variable(var_name, shape, tf.float32, initializer = tf.zeros_initializer)
	return b

class DQN:

	def __init__(self, network_name):
		with tf.variable_scope(network_name):
			self.w1, self.b1 = weight_initializer([1600, 512], 'w1'), bias_initializer([512], 'b1')
			self.w2, self.b2 = weight_initializer([512, 3], 'w2'), bias_initializer([3], 'b2')

	def forward(self, X, network_name):
		with tf.variable_scope(network_name):
			h1 = tf.matmul(X, self.w1, name='h1') + self.b1
			a1 = tf.nn.relu(h1, name='a1')
			q_values = tf.matmul(a1, self.w2, name='q_values') + self.b2
			normalized_q_values = tf.nn.softmax(q_values, name='normalized_q_values')
			return normalized_q_values

def train(episodes):
	episode = 1
	replay = ReplayBuffer()
	epsilon = 1
	epsilon_decay = 0.99
	discount = 0.9

	X = tf.placeholder(shape=(None, 1600), dtype=tf.float32)
	Y = tf.placeholder(shape=(None, 3), dtype=tf.float32)

	cur_model, target_model = DQN('cur_model'), DQN('target_model')
	current_q_values, target_q_values = cur_model.forward(X, 'cur_model'), target_model.forward(X, 'target_model')
	loss = tf.losses.softmax_cross_entropy(Y, current_q_values)
	optimizer = tf.train.AdamOptimizer()
	model = optimizer.minimize(loss)

	with tf.compat.v1.Session() as sess:
		# First we have to initialize all the variables present in our computational graph
		sess.run(tf.compat.v1.global_variables_initializer())
		while episode <= episodes:
			time = 1
			while time < 6:
				prev_state, cur_state, next_state = None, None, None
				if uniform() < epsilon:
					action = randint(3)
				else:
					img = cv2.imread('cur_state.jpg')
					cur_state = preprocess_frame(cur_state.reshape((-1, 1600)), prev_state)
					q_values = sess.run(target_q_values, feed_dict = {X : img.reshape(-1, 1600)})
					action = tf.argmax(q_values, axis=1)
				next_state, reward, end = game.play(action)
				replay.add_to_memory(cur_state, action, next_state, reward, end)
				if end == 1:
					time += 1

			# Training

if __name__ == '__main__':
	train()