import tensorflow.dtypes as tf_dtype

from tensorflow import multiply, reduce_sum, reduce_mean, name_scope, one_hot
from tensorflow.compat.v1 import (disable_eager_execution, reset_default_graph, trainable_variables, 
	global_variables_initializer, Session, placeholder)
from tensorflow.compat.v1.summary import FileWriter, scalar, merge_all
from tensorflow.compat.v1.train import Saver, AdamOptimizer
from tensorflow.math import squared_difference
from dqn import DQNModel
from pong import Pong
from numpy.random import uniform, choice
from numpy import argmax, reshape
from memory import ReplayBuffer
from preprocessing import prepare_input
from util import skip_initial_frames, cal_correct_q_values, test_model
from tqdm import tqdm

# Hyper-parameters
epsilon = 1
LEARNING_RATE = 0.0001
EPSILON_DECAY = 0.01
EPSILON_DECAY_FREQ = 200
ITERATIONS = int(1e4)
TIMESTEPS = 256
EPOCHS = 8
BATCH_SIZE = 32
DISCOUNT = 0.9
MODEL_SAVING_FREQ = 5e3
N_ACTIONS = 2

def train():
	global epsilon
	disable_eager_execution()
	# It is safe to clear the computation graph because there still could be variables present from the previous run  
	reset_default_graph()
	# Creating summary object
	writer = FileWriter("./summary/")

	# Placeholders
	inputs = placeholder(shape=(None, 80, 80, 4), dtype=tf_dtype.float32, name="inputs")
	correct_q_values_ph = placeholder(shape=(BATCH_SIZE,), dtype=tf_dtype.float32, name="correct_q_values")
	action_indexes = placeholder(shape=(BATCH_SIZE,), dtype=tf_dtype.int32, name="action_indexes")
	reward_history = placeholder(shape=(), dtype=tf_dtype.float32, name="reward_history")
	loss_history = placeholder(shape=(), dtype=tf_dtype.float32, name="loss_history")

	# Metrics to be displayed on tensorboard
	scalar("Rewards", reward_history, collections=["Rewards"], family="Rewards")
	scalar("Loss", loss_history, collections=["Losses"], family="Losses")
	loss_summary = merge_all(key="Losses", name="loss_summary")
	rewards_summary = merge_all(key="Rewards", name="rewards_summary")
	# Initializing memory buffer
	replay = ReplayBuffer()
	# Initializing the network
	dqn_model = DQNModel()
	# Getting Q-values from the network
	q_layer = dqn_model.forward(inputs)
	predicted_q_values = reduce_sum(multiply(q_layer, one_hot(action_indexes, N_ACTIONS)), axis=1, name="predicted_q_values")
	
	# Loss function and optimizer initialization
	with name_scope("loss"):
		loss = reduce_mean(squared_difference(predicted_q_values, correct_q_values_ph))
	with name_scope("optimizer"):
		optimizer = AdamOptimizer(learning_rate=LEARNING_RATE, name="adam")
		minimize_loss = optimizer.minimize(loss, var_list=trainable_variables(scope="dqn/weights_bias"), name="minimize_loss")

	with Session() as sess:
		# First we have to initialize all the variables present in our computational graph
		sess.run(global_variables_initializer())
		# Object for model saving
		saver = Saver()
		# For visualizing graph on tensorboard
		writer.add_graph(sess.graph)
		game = Pong()
		# Skipping some initial frames so that the ball and the opponent paddle come in the view
		cur_frame = skip_initial_frames(game)
		cur_state = prepare_input(cur_frame)
		max_q_index, v_value, action = None, None, None

		# Training begins
		for itr in tqdm(range(1, ITERATIONS+1)):
			rewards, step = 0, 1
			while step <= TIMESTEPS:
				q_values = sess.run(q_layer, feed_dict = {inputs : reshape(cur_state, (1, 80, 80, 4))})
				v_value = max(q_values[0])
				# Deciding if exploration/exploitation should be performed
				if uniform() < epsilon:
					action = choice([2, 3])
					max_q_index = 0 if action == 2 else 1
				else:
					max_q_index = argmax(q_values[0])
					action = 2 if max_q_index == 0 else 3
				# Performing action in the game
				next_frame, reward = game.play(action)
				rewards += reward
				if reward != 0:
					replay.add_to_memory(cur_state, reward, max_q_index, reward, True)
					# Skipping some initial frames so that the ball and the opponent paddle come in the view
					cur_frame = skip_initial_frames(game)
					cur_state = prepare_input(cur_frame)
				else:
					replay.add_to_memory(cur_state, v_value, max_q_index, reward, False)
					# Updating next state
					cur_state = prepare_input(next_frame, cur_state)
				step += 1
			
			correct_q_values = cal_correct_q_values(replay.rewards, replay.v_values, replay.are_terminals, DISCOUNT)
			model_loss = 0
			# Will train our current model in batches
			for _ in range(EPOCHS):
				batch_states, batch_actions_taken, batch_correct_q_values = replay.batch(BATCH_SIZE, correct_q_values)
				_, cur_loss = sess.run((minimize_loss, loss), feed_dict={inputs: batch_states, action_indexes: batch_actions_taken,
					correct_q_values_ph: batch_correct_q_values})
				model_loss += cur_loss
			# Clearing memory buffer for next iteration
			replay.clear()
			summ = sess.run(loss_summary, feed_dict={loss_history: model_loss/EPOCHS})
			writer.add_summary(summ, itr)
			summ = sess.run(rewards_summary, feed_dict={reward_history: rewards})
			writer.add_summary(summ, itr)

			# Decaying exploration rate over time
			if itr % EPSILON_DECAY_FREQ == 0 and epsilon > 0.2:
				epsilon -= EPSILON_DECAY
			
			# Saving model after every fixed iterations
			if itr%MODEL_SAVING_FREQ == 0:
				saver.save(sess, './model/dqn', global_step=itr)
				test_model(sess, inputs, q_layer, game)

		game.close()

if __name__ == '__main__':
	train()