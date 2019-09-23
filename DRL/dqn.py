import tensorflow as tf
import gym
import time
import cv2
from numpy.random import uniform, choice
from numpy import zeros, argmax, stack, append, reshape
from memory import ReplayBuffer
from preprocessing import preprocess_frame, show_img
from tqdm import tqdm

xavier_initializer = tf.glorot_normal_initializer()

def weights_initialize(shape, var_name):
	w = tf.Variable(xavier_initializer(shape=shape, dtype=tf.float32), name=var_name, shape=shape, dtype=tf.float32)
	return w

def bias_initialize(shape, var_name):
	b = tf.Variable(tf.zeros(shape=shape, dtype=tf.float32), name=var_name, shape=shape, dtype=tf.float32)
	return b

def conv2D(inputs, kernel, stride, pad, bias, conv_layer, act_layer):
	conv = tf.compat.v1.nn.bias_add(tf.compat.v1.nn.conv2d(inputs, filter=kernel, strides=stride, padding=pad), bias, name=conv_layer)
	act = tf.compat.v1.nn.relu(conv, name=act_layer)
	return act

def pool2D(inputs, k_size, stride, pad, pool_layer):
	pool = tf.compat.v1.nn.max_pool2d(inputs, k_size, stride, pad, name=pool_layer)
	return pool 

def fc_layer(prev_layer, weights, bias, non_linearity, fc_layer, act_layer=None):
	out = tf.compat.v1.nn.bias_add(tf.matmul(prev_layer, weights), bias, name=fc_layer)
	if non_linearity == 'relu':
		act = tf.compat.v1.nn.relu(out, name=act_layer)
		return act
	elif non_linearity == 'none':
		return out
	else:
		raise Exception('Wrong function for non-linearity')

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
	next_frame = None
	for i in range(20):
		next_frame, reward = game.play(0)

	return preprocess_frame(next_frame)

def test_model(sess, X, current_q_values, game):
	"""
	This function will test our trained model
	"""

	# four character code object for video writer
	ex = cv2.VideoWriter_fourcc('M','J','P','G')
	# video writer object
	out = cv2.VideoWriter("./test/test_{}.avi".format(time.ctime()), ex, 5.0, (160, 210))
	
	model_point, computer_point = 0, 0
	cur_frame = skip_initial_frames(game)
	# Stacking 4 frames to capture motion
	cur_state = stack((cur_frame, cur_frame, cur_frame, cur_frame), axis=2)
	cur_state = reshape(cur_state, (80, 80, 4))
	while model_point < 13 and computer_point < 13:
		q_values = sess.run(current_q_values, feed_dict = {X : reshape(cur_state, (1, 80, 80, 4))})
		max_q_index = argmax(q_values)
		action = 2 if max_q_index == 0 else 3
		next_frame, reward = game.play(action)
		# write frame to video writer
		out.write(next_frame)
		if reward == 1 or reward == -1:
			if reward == -1:
				computer_point += 1
			else:
				model_point += 1
			cur_frame = skip_initial_frames(game)
			cur_state = stack((cur_frame, cur_frame, cur_frame, cur_frame), axis=2)
			cur_state = reshape(cur_state, (80, 80, 4))
		else:
			cur_state = append(preprocess_frame(next_frame), cur_state[:, :, 0:3], axis=2)

	out.release()

class DQN:

	def __init__(self, network_name):
		with tf.compat.v1.variable_scope(network_name):
			# Filters for first layer of convolution 
			self.w1, self.b1 = weights_initialize([3, 3, 4, 32], 'w1'), bias_initialize([32], 'b1')
			# Filters for second layer of convolution 
			self.w2, self.b2 = weights_initialize([3, 3, 32, 64], 'w2'), bias_initialize([64], 'b2')
			# Weights & Bias for the first FC layer
			self.w3, self.b3 = weights_initialize([6400, 512], 'w3'), bias_initialize([512], 'b3')
			# Weights & Bias for the final FC layer / Q-layer
			self.w4, self.b4 = weights_initialize([512, 2], 'w4'), bias_initialize([2], 'b4')
			self.tensors = {'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2, 'w3': self.w3, 'b3': self.b3, 'w4': self.w4, 'b4': self.b4}

	def forward(self, X, network_name):
		with tf.compat.v1.variable_scope(network_name):
			# First layer of convolution
			conv1 = conv2D(X, self.w1, 2, 'SAME', self.b1, 'conv1', 'act_1')
			# Second layer of convolution
			conv2 = conv2D(conv1, self.w2, 2, 'SAME', self.b2, 'conv2', 'act_2')
			pool2 = pool2D(conv2, 2, 2, 'SAME', 'pool_2')
			# Flattening out the previous layer
			flatten = tf.reshape(pool2, [-1, 6400])
			# First FC layer
			fc1 = fc_layer(flatten, self.w3, self.b3, 'relu', 'fc1', 'act_3')
			# Final layer with Q-values
			q_values = fc_layer(fc1, self.w4, self.b4, 'none', 'q_values')
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
		return next_state, reward

	def get_all_actions(self):
		return self.action_mapping

def train(episodes, max_steps):
	# It is safe to clear the computation graph because there still could be variables present from the previous run  
	tf.compat.v1.reset_default_graph()
	# Creating summary object
	writer = tf.compat.v1.summary.FileWriter('./graph/')

	# Hyper-parameters
	epsilon = 1
	learning = 0.0001
	epsilon_decay = 0.001
	iterations = 100
	batch_size = 32
	discount = 0.9

	X = tf.compat.v1.placeholder(shape=(None, 80, 80, 4), dtype=tf.float32)
	Y = tf.compat.v1.placeholder(shape=(None, 2), dtype=tf.float32)
	reward_history = tf.compat.v1.placeholder(shape=(), dtype=tf.int32)
	# For showing rewards earned after every 10 episodes in tensorboard
	tf.compat.v1.summary.scalar('Rewards', reward_history, family='Rewards')
	# Initializing memory buffer
	replay = ReplayBuffer()
	# Initializing network models
	cur_model, target_model = DQN('cur_model'), DQN('target_model')
	# Getting Q-values from network models
	current_q_values, target_q_values = cur_model.forward(X, 'cur_model'), target_model.forward(X, 'target_model')
	# Loss function and optimizer initialization
	loss = tf.reduce_mean(tf.compat.v1.squared_difference(Y, current_q_values))
	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning)
	minimize_loss = optimizer.minimize(loss)
	summary = tf.compat.v1.summary.merge_all()

	with tf.compat.v1.Session() as sess:
		# First we have to initialize all the variables present in our computational graph
		sess.run(tf.compat.v1.global_variables_initializer())
		# Object for model saving
		saver = tf.compat.v1.train.Saver()
		# For visualizing graph on tensorboard
		writer.add_graph(sess.graph)
		game = Pong()
		for episode in tqdm(range(1, episodes+1)):
			step, rewards = 0, 0
			# Skipping some initial frames so that the ball and the opponent paddle come in the view
			cur_frame = skip_initial_frames(game)
			# Stacking 4 frames to capture motion
			cur_state = stack((cur_frame, cur_frame, cur_frame, cur_frame), axis=2)
			cur_state = reshape(cur_state, (80, 80, 4))
			while step <= max_steps:
				# Deciding if exploration/exploitation should be performed
				if uniform() < epsilon:
					action = choice([2, 3])
					max_q_index = 0 if action == 2 else 1
				else:
					q_values = sess.run(current_q_values, feed_dict = {X : reshape(cur_state, (1, 80, 80, 4))})
					max_q_index = argmax(q_values)
					action = 2 if max_q_index == 0 else 3

				# Performing action in the game
				next_frame, reward = game.play(action)
				# Updating next state
				next_state = append(preprocess_frame(next_frame), cur_state[:, :, 0:3], axis=2)
				rewards += reward
				replay.add_to_memory(cur_state, max_q_index, next_state, reward)
				if reward in [-1, 1]:
					# Skipping some initial frames so that the ball and the opponent paddle come in the view
					cur_frame = skip_initial_frames(game)
					# Stacking 4 frames to capture motion
					cur_state = stack((cur_frame, cur_frame, cur_frame, cur_frame), axis=2)
					cur_state = reshape(cur_state, (80, 80, 4))
				else:
					cur_state = next_state
				
				step += 1

			# Training begins with generation of labels
			for i in range(iterations):
				y = zeros((batch_size, 2))
				x = zeros((batch_size, 80, 80, 4))
				for index, composed_state in enumerate(replay.batch(batch_size)):
					cur_state, action, next_state, reward = composed_state
					# Getting Q-values from the current network for current state
					cur_q_labels = sess.run(current_q_values, feed_dict = {X : reshape(cur_state, (1, 80, 80, 4))})
					# Getting Q-values from the target network for next state
					target_q_labels = sess.run(target_q_values, feed_dict = {X : reshape(next_state, (1, 80, 80, 4))})
					if reward in [-1, 1]:
						cur_q_labels[0, action] = reward
					else:
						cur_q_labels[0, action] = reward + discount * target_q_labels[0, argmax(target_q_labels)]

					x[index] = cur_state
					y[index] = cur_q_labels

				# Minimizing loss / Performing Q-learning
				loss = sess.run(minimize_loss, feed_dict = {X: x, Y: y})

			# Clearing memory buffer for next episode
			replay.clear()

			# Decaying exploration rate after every 30 episodes
			if episode % 30 == 0 and epsilon > 0.15:
				epsilon -= epsilon_decay

			# Updating target network after every episode
			update_target_network(sess, cur_model, target_model)

			# Saving model after every 500 episodes
			if episode%500 == 0:
				saver.save(sess, './model/dqn', global_step=episode)
			
			# Will test our model after every 100 episodes
			if episode%100 == 0:
				test_model(sess, X, current_q_values, game)
			
			# Will see the rewards earned after every 10 episode
			if episode%10 == 0:
				print('Reward earned after {} episode is: {}'.format(episode, rewards))
				summ = sess.run(summary, feed_dict={reward_history: rewards})
				writer.add_summary(summ, episode)

		game.close()

def sanity_checking():
	game = Pong()
	cur_frame = skip_initial_frames(game)
	while True:
		cur_state = stack((cur_frame, cur_frame, cur_frame, cur_frame), axis=2)
		next_frame, reward = game.play(choice([2, 3]))
		prev_frame = cur_frame
		cur_frame = preprocess_frame(next_frame)
		ch = raw_input('Continue?')
		if ch == 'n':
			break

if __name__ == '__main__':
	train(5000, 10000)