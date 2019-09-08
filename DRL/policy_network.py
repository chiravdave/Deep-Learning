import tensorflow as tf
import gym
import time
from numpy.random import uniform, randint
from memory import ReplayBuffer
from preprocessing import preprocess_frame

def weights_initialize(shape, var_name):
	w = tf.compat.v1.Variable(var_name, shape, tf.float32, initializer = tf.contrib.layers.xavier_initializer())
	return w

def bias_initialize(shape, var_name):
	b = tf.compat.v1.Variable(var_name, shape, tf.float32, initializer = tf.zeros_initializer)
	return b

def copy_network_weights(source_network, target_network):
	pass

class PolicyNetwork:

	def __init__(self, network_name):
		with tf.variable_scope(network_name):
			self.w1, self.b1 = weight_initializer([1600, 512], 'w1'), bias_initializer([512], 'b1')
			self.w2, self.b2 = weight_initializer([512, 1], 'w2'), bias_initializer([1], 'b2')

	def forward(self, X, network_name):
		with tf.variable_scope(network_name):
			h1 = tf.matmul(X, self.w1, name='h1') + self.b1
			a1 = tf.nn.relu(h1, name='a1')
			h2 = tf.matmul(a1, self.w2, name='h2') + self.b2
			policy = tf.nn.sigmoid(h2, name='policy')
			return policy

class Pong:

	def __init__(self):
		self.env = gym.make('Pong-v0')
		self.action_mapping = {'up': 2, 'down': 3, 'stay': 0}

	def reset(self):
		return self.env.reset()

	def show(self):
		self.env.render()

	def close(self):
		self.env.close()
		self.env = None

	def play(self, action):
		next_state, reward, is_done, _ = self.env.step(action)
		return next_state, reward, is_done

	def get_action_mapping(self):
		return self.action_mapping

def train(episodes, max_game_points):
	episode = 1
	replay = ReplayBuffer()
	epsilon = 1
	learn_rate = 0.001
	epsilon_decay = 0.99
	discount = 0.9

	X = tf.placeholder(shape=(None, 1600), dtype=tf.float32)
	Y = tf.placeholder(shape=(None, 1), dtype=tf.float32)

	cur_model, target_model = PolicyNetwork('cur_model'), PolicyNetwork('target_model')
	cur_policy, target_policy = cur_model.forward(X, 'cur_model'), target_model.forward(X, 'target_model')
	with tf.name_scope('cur_model'):
		loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=cur_policy, labels=Y, name='loss')
		optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate, name='adam')
		model = optimizer.minimize(loss)

	with tf.compat.v1.Session() as sess:
		# First we have to initialize all the variables present in our computational graph
		sess.run(tf.compat.v1.global_variables_initializer())
		game = Pong()
		while episode <= episodes:
			game_point = 0
			cur_state = preprocess_frame(game.reset())
			prev_state = None
			while game_point < max_game_points:
				game.show()
				x = cur_state - prev_state if prev_state else np.zeros(80*80)
				prev_state = np.copy(x)
				if uniform() < epsilon:
					action = randint(low=2, high=4)
				else:
					policy = sess.run(cur_policy, feed_dict = {X : x.reshape(-1, 1600)})
					action = 2 if policy >= 0.5 else 3
				next_state, reward, end = game.play(action)
				pass
				replay.add_to_memory(x, action, next_state, reward, end)
				if end == 1:
					time += 1

			# Training

if __name__ == '__main__':
	train(10, 10)