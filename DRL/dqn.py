import tensorflow as tf
import gym
from numpy.random import uniform, choice
from numpy import zeros, argmax
import time
from memory import ReplayBuffer
from preprocessing import preprocess_frame
from tqdm import tqdm

xavier_initializer = tf.glorot_normal_initializer()

def weights_initialize(shape, var_name):
	w = tf.Variable(xavier_initializer(shape=shape, dtype=tf.float32), name=var_name, shape=shape, dtype=tf.float32)
	return w

def bias_initialize(shape, var_name):
	b = tf.Variable(tf.zeros(shape=shape, dtype=tf.float32), name=var_name, shape=shape, dtype=tf.float32)
	return b

def update_target_network(sess, source_network, target_network):
	"""
	This function will update the target network.
	"""

	for key, tensor in source_network.tensors.items():
		sess.run(target_network.tensors[key].assign(tensor))

def skip_initial_frames(game):
	"""
	This function will skip some initial frames so that the ball and the opponent paddle comes in the view
	"""

	game.reset()
	prev_state, cur_state = None, None
	for i in range(20):
		action = choice([2, 3])
		next_state, reward, end = game.play(action)
		if i == 18:
			prev_state = next_state
		elif i == 19:
			cur_state = next_state

	return preprocess_frame(cur_state) - preprocess_frame(prev_state)

class DQN:

	def __init__(self, network_name):
		with tf.compat.v1.variable_scope(network_name):
			self.w1, self.b1 = weights_initialize([6400, 512], 'w1'), bias_initialize([512], 'b1')
			self.w2, self.b2 = weights_initialize([512, 2], 'w2'), bias_initialize([2], 'b2')
			self.tensors = {'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2}

	def forward(self, X, network_name):
		with tf.compat.v1.variable_scope(network_name):
			h1 = tf.matmul(X, self.w1, name='h1') + self.b1
			a1 = tf.nn.relu(h1, name='a1')
			q_values = tf.matmul(a1, self.w2, name='q_values') + self.b2
			return q_values

class Pong:

	def __init__(self):
		self.env = gym.make('Pong-v0')
		self.action_mapping = {'up': 2, 'down': 3}

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
	# It is safe to clear the computation graph because there still could be variables present from the previous run  
	tf.compat.v1.reset_default_graph()
	# Creating summary object
	writer = tf.compat.v1.summary.FileWriter('./graph/')

	# Hyper-parameters
	epsilon = 1
	learning = 0.001
	epsilon_decay = 0.98
	iterations = 1000
	batch_size = 32
	discount = 0.9

	X = tf.compat.v1.placeholder(shape=(None, 6400), dtype=tf.float32)
	Y = tf.compat.v1.placeholder(shape=(None, 2), dtype=tf.float32)
	rewards = tf.compat.v1.placeholder(shape=(), dtype=tf.float32)
	# For showing average rewards earned per episode in tensorboard
	tf.compat.v1.summary.scalar('Rewards', rewards, family='Rewards')
	# Initializing memory buffer
	replay = ReplayBuffer()
	# Initializing network models
	cur_model, target_model = DQN('cur_model'), DQN('target_model')
	# Getting Q-values from network models
	current_q_values, target_q_values = cur_model.forward(X, 'cur_model'), target_model.forward(X, 'target_model')

	# Loss function and optimizer initialization
	loss = tf.reduce_mean(tf.compat.v1.squared_difference(Y, current_q_values))
	optimizer = tf.train.AdamOptimizer(learning_rate = learning)
	minimize_loss = optimizer.minimize(loss)
	summary = tf.compat.v1.summary.merge_all()

	with tf.compat.v1.Session() as sess:
		# First we have to initialize all the variables present in our computational graph
		sess.run(tf.compat.v1.global_variables_initializer())
		# Object for model saving
		saver = tf.compat.v1.train.Saver()
		game = Pong()
		for episode in tqdm(range(episodes)):
			step, avg_reward = 0, 0
			# Skipping some initial frames so that the ball and the opponent paddle come in the view
			cur_state = skip_initial_frames(game) 
			while step <= max_steps:
				# Visualizing next move in the game
				game.show()
				#time.sleep(1)
				# Deciding if exploration/exploitation should be performed
				if uniform() < epsilon:
					action = choice([2, 3])
					max_q_index = 0 if action == 2 else 1
				else:
					
					q_values = sess.run(current_q_values, feed_dict = {X : x})
					max_q_index = argmax(q_values)
					action = 2 if max_q_index == 0 else 3

				# Performing action in the game
				next_state, reward, end = game.play(action)
				avg_reward += reward
				# Taking difference between current and older frame to capture ball motion
				next_state = preprocess_frame(next_state) - cur_state
				replay.add_to_memory(cur_state, max_q_index, next_state, reward, end)
				if end:
					# Skipping some initial frames so that the ball and the opponent paddle come in the view
					cur_state = skip_initial_frames(game)
				else:
					cur_state = next_state
				
				step += 1

			# Calculating average reward for this episode
			avg_reward /= max_steps
			summ = sess.run(summary, feed_dict={rewards: avg_reward})
			writer.add_summary(summ, episode)

			# Training begins with generation of labels
			for i in range(iterations):
				y = zeros((batch_size, 2))
				x = zeros((batch_size, 6400))
				for index, composed_state in enumerate(replay.batch(batch_size)):
					cur_state, action, next_state, reward, end = composed_state
					# Getting Q-values from the current network for current state
					cur_q_labels = sess.run(current_q_values, feed_dict = {X : cur_state})
					# Getting Q-values from the target network for next state
					next_q_labels = sess.run(target_q_values, feed_dict = {X : next_state})
					
					if end:
						cur_q_labels[0, action] = reward
					else:
						cur_q_labels[0, action] = reward + discount * next_q_labels[0, argmax(next_q_labels)]

					x[index] = cur_state
					y[index] = cur_q_labels

				# Minimizing loss / Performing Q-learning
				loss = sess.run(minimize_loss, feed_dict = {X: x, Y: y})

			# Clearing memory buffer for next episode
			replay.clear()

			# Decaying exploration rate after every 200 episodes
			if episode % 200 == 0:
				epsilon *= epsilon_decay

			# Updating target network after every 50 episodes
			if episode % 80 == 0:
				update_target_network(sess, cur_model, target_model)

			# Saving model after every 500 episodes
			if episode%500 == 0:
				saver.save(sess, './model/gan_model', global_step=episode)

		game.close()
		# For visualizing graph on tensorboard
		writer.add_graph(sess.graph)

if __name__ == '__main__':
	train(1000, 70000)