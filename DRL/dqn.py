import tensorflow as tf
import gym
from numpy.random import uniform, randint
from memory import ReplayBuffer
from preprocessing import preprocess_frame

def weights_initialize(shape, var_name):
	w = tf.compat.v1.Variable(var_name, shape, tf.float32, initializer = tf.contrib.layers.xavier_initializer())
	return w

def bias_initialize(shape, var_name):
	b = tf.compat.v1.Variable(var_name, shape, tf.float32, initializer = tf.zeros_initializer)
	return b

def update_target_network(source_network, target_network):
	pass

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
			return q_values

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

	def get_all_actions(self):
		return self.action_mapping

def train(episodes, max_steps):
	episode = 1
	replay = ReplayBuffer()
	epsilon = 1
	epsilon_decay = 0.99
	discount = 0.9
	game_action_mapping = {0: 2, 1: 3, 2: 0}

	X = tf.placeholder(shape=(None, 1600), dtype=tf.float32)
	Y = tf.placeholder(shape=(None, 3), dtype=tf.float32)

	cur_model, target_model = DQN('cur_model'), DQN('target_model')
	current_q_values, target_q_values = cur_model.forward(X, 'cur_model'), target_model.forward(X, 'target_model')
	loss = tf.reduce_mean(tf.compat.v1.squared_difference(Y, current_q_values))
	optimizer = tf.train.AdamOptimizer()
	minimize_loss = optimizer.minimize(loss)

	with tf.compat.v1.Session() as sess:
		# First we have to initialize all the variables present in our computational graph
		sess.run(tf.compat.v1.global_variables_initializer())
		game = Pong()
		while episode <= episodes:
			step = 0
			prev_state = None
			cur_state = preprocess_frame(game.reset())
			while step <= max_steps:
				# Visualizing next move in the game
				game.show()
				x = cur_state - prev_state if prev_state else np.zeros(80*80)
				if uniform() < epsilon:
					action = randint(3)
				else:
					
					q_values = sess.run(current_q_values, feed_dict = {X : img.reshape(-1, 1600)})
					max_q_index = tf.argmax(q_values, axis=1)
					action = game_action_mapping[max_q_index]
				next_state, reward, end = game.play(action)
				next_state = preprocess_frame(next_state)
				replay.add_to_memory(cur_state, max_q_index, next_state, reward, end)
				if end:
					prev_state = None
					cur_state = preprocess_frame(game.reset())
				else:
					prev_state = cur_state
					cur_state = next_state
					step += 1

			# Training
			y = np.zeros((32, 3))
			x = np.empty((32, 1600))
			for index, cur_state, max_q_index, next_state, reward, end in enumerate(replay.batch(32)):
				q_values = sess.run(current_q_values, feed_dict = {X : img.reshape(-1, 1600)})
				if end:
					q_values[0, max_q_index] = reward
				else:
					t_q_values = sess.run(target_q_values, feed_dict = {X : img.reshape(-1, 1600)})
					q_values[0, max_q_index] = reward + discount * tf.max(t_q_values, axis=1)
				x[index] = cur_state
				y[index] = q_values

			loss = sess.run(minimize_loss, feed_dict = {X: x, Y: y})

			episode += 1

		game.close()

if __name__ == '__main__':
	pong = Pong()
	pong.show()
	print(pong.get_all_actions())
	raw_input()
	pong.close()
	#train(10, 10)