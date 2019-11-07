import tensorflow as tf
import time
import cv2

from pong import Pong
from layers import weights_initialize, bias_initialize, conv2D, pool2D, fc_layer
from numpy.random import uniform
from numpy import zeros, stack, append, reshape
from preprocessing import preprocess_frame
from tqdm import tqdm

def skip_initial_frames(game):
	"""
	This function will skip some initial frames so that the ball and the opponent paddle comes in the view.

	:param game: pong game object
	:rtype: processed frame which will be used as the starting frame to train our PG Agent
	"""

	game.reset()
	next_frame = None
	for i in range(20):
		next_frame, reward = game.play(0)

	return preprocess_frame(next_frame)

def test_model(sess, X, policy, game):
	"""
	This function will test our trained model. It will create a video for one round (13 points) of the game.

	:param sess: session object
	:param X: input placeholder
	:param policy: policy from the network
	:param game: pong game object
	"""

	# four character code object for video writer
	ex = cv2.VideoWriter_fourcc('M','J','P','G')
	# video writer object
	out = cv2.VideoWriter("./test/pg_agent_{}.avi".format(time.ctime()), ex, 5.0, (160, 210))
	
	model_point, computer_point = 0, 0
	cur_frame = skip_initial_frames(game)
	# Stacking 4 frames to capture motion
	cur_state = stack((cur_frame, cur_frame, cur_frame, cur_frame), axis=3)
	cur_state = reshape(cur_state, (80, 80, 4))

	# Game begins
	while model_point < 13 and computer_point < 13:
		cur_policy = sess.run(policy, feed_dict = {X : reshape(cur_state, (1, 80, 80, 4))})
		action = 2 if cur_policy[0][0] >= 0.5 else 3
		next_frame, reward = game.play(action)
		# write frame to video writer
		out.write(next_frame)
		if reward != 0:
			if reward == -1:
				computer_point += 1
			else:
				model_point += 1
			cur_frame = skip_initial_frames(game)
			cur_state = stack((cur_frame, cur_frame, cur_frame, cur_frame), axis=3)
			cur_state = reshape(cur_state, (80, 80, 4))
		else:
			cur_state = append(preprocess_frame(next_frame), cur_state[:, :, 0:3], axis=2)

	out.release()

def get_discounted_rewards(labels, rewards, discount):
	n_samples = len(rewards)
	discounted_rewards = zeros((n_samples, 1))
	running_reward = 0
	for end in range(n_samples-1, -1, -1):
		if rewards[end] != 0:
			running_reward = 0

		running_reward = running_reward * discount + rewards[end] 
		discounted_rewards[end][0] = abs(running_reward)
		
		# Checking if the action taken was bad
		if running_reward < 0:
			# Flipping labels
			labels[end][0] = 1 - labels[end][0]

	return discounted_rewards

class PGAgent:

	def __init__(self, network_name):
		with tf.compat.v1.variable_scope(network_name):
			# Filters for first layer of convolution 
			self.w1, self.b1 = weights_initialize([3, 3, 4, 32], 'w1'), bias_initialize([32], 'b1')
			# Filters for second layer of convolution 
			self.w2, self.b2 = weights_initialize([3, 3, 32, 64], 'w2'), bias_initialize([64], 'b2')
			# Weights & Bias for the first FC layer
			self.w3, self.b3 = weights_initialize([6400, 512], 'w3'), bias_initialize([512], 'b3')
			# Weights & Bias for the final FC layer / Policy layer
			self.w4, self.b4 = weights_initialize([512, 1], 'w4'), bias_initialize([1], 'b4')

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
			# Policy layer with Action values
			policy = fc_layer(fc1, self.w4, self.b4, 'sigmoid', 'policy')
			return policy

def train(episodes):
	# It is safe to clear the computation graph because there still could be variables present from the previous run  
	tf.compat.v1.reset_default_graph()
	# Creating summary object
	writer = tf.compat.v1.summary.FileWriter('./graph/')

	# Hyper-parameters
	learning = 0.0001
	batch_size = 128
	discount = 0.9

	# Placeholder for inputs, outputs, rewards and gradient weights
	X = tf.compat.v1.placeholder(shape=(None, 80, 80, 4), dtype=tf.float32)
	Y = tf.compat.v1.placeholder(shape=(None, 1), dtype=tf.float32)
	gradient_weights = tf.compat.v1.placeholder(shape=(None, 1), dtype=tf.float32)
	reward_history = tf.compat.v1.placeholder(shape=(), dtype=tf.int32)

	# For showing rewards earned after every 10 episodes in tensorboard
	tf.compat.v1.summary.scalar('Rewards', reward_history, family='Rewards')

	# Initializing policy gradient model
	policy_network = PGAgent('Policy_Network')
	policy = policy_network.forward(X, 'Policy_Network')
	# Loss function and optimizer initialization
	loss = tf.reduce_mean(tf.compat.v1.losses.log_loss(Y, policy, weights=gradient_weights))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning, name='adam')
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

		# Training begins
		for episode in tqdm(range(1, episodes+1)):
			n_samples, total_rewards = 0, 0
			# Initializing memory buffer for rewards
			rewards = []
			# Skipping some initial frames so that the ball and the opponent paddle come in the view
			cur_frame = skip_initial_frames(game)
			# Stacking 4 frames to capture motion
			cur_state = stack((cur_frame, cur_frame, cur_frame, cur_frame), axis=3)
			cur_state = reshape(cur_state, (80, 80, 4))
			xs = zeros((batch_size, 80, 80, 4))
			ys = zeros((batch_size, 1))
			while n_samples < batch_size:
				cur_policy = sess.run(policy, feed_dict = {X : reshape(cur_state, (1, 80, 80, 4))})
				# Deciding if exploration/exploitation should be performed
				if uniform() < cur_policy[0][0]:
					action = 2
					ys[n_samples][0] = 1

				else:
					action = 3
					ys[n_samples][0] = 0

				# Performing action in the game
				next_frame, reward = game.play(action)
				# print(cur_policy, reward)
				# Updating next state
				next_state = append(preprocess_frame(next_frame), cur_state[:, :, 0:3], axis=2)
				total_rewards += reward
				xs[n_samples] = cur_state
				rewards.append(reward)
				if reward != 0:
					# Skipping some initial frames so that the ball and the opponent paddle come in the view
					cur_frame = skip_initial_frames(game)
					# Stacking 4 frames to capture motion
					cur_state = stack((cur_frame, cur_frame, cur_frame, cur_frame), axis=3)
					cur_state = reshape(cur_state, (80, 80, 4))
				else:
					cur_state = next_state

				n_samples += 1

			# Adjusting error with the Advantage function
			discounted_rewards = get_discounted_rewards(ys, rewards, discount)

			# Minimizing loss
			sess.run(minimize_loss, feed_dict = {X: xs, Y: ys, gradient_weights: discounted_rewards})

			# Will test our model after every 100 episodes
			if episode%2500 == 0:
				test_model(sess, X, policy, game)

			# Will see the rewards earned after every 10 episode
			if episode%10 == 0:
				print('Reward earned after {} episode is: {}'.format(episode, total_rewards))
				summ = sess.run(summary, feed_dict={reward_history: total_rewards})
				writer.add_summary(summ, episode)

			# Saving model after every 500 episodes
			if episode%10000 == 0:
				saver.save(sess, './model/pg_agent', global_step=episode)
				batch_size *= 2

		game.close()

if __name__ == '__main__':
	train(30000)