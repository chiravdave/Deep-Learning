import cv2
import time

from numpy import array, float32, argmax, reshape
from preprocessing import prepare_input

def test_model(sess, inputs, q_layer, game):
	"""
	This function will test our trained model. It will create a video for one round (13 points) of the game.

	:param sess: session object
	:param inputs: input placeholder
	:param q_layer: variable responsible for getting the Q-values from the current network
	:param game: pong game object
	"""

	# four character code object for video writer
	ex = cv2.VideoWriter_fourcc('M','J','P','G')
	# video writer object
	out = cv2.VideoWriter("./test/test_{}.avi".format(time.ctime()), ex, 5.0, (160, 210))
	
	model_point, computer_point = 0, 0
	cur_frame = skip_initial_frames(game)
	cur_state = prepare_input(cur_frame)

	# Game begins
	while model_point < 13 and computer_point < 13:
		q_values = sess.run(q_layer, feed_dict = {inputs : reshape(cur_state, (1, 80, 80, 4))})
		max_q_index = argmax(q_values[0])
		action = 2 if max_q_index == 0 else 3
		next_frame, reward = game.play(action)
		# write frame to video writer
		out.write(next_frame)
		if reward != 0:
			if reward == -1:
				computer_point += 1
			else:
				model_point += 1
			cur_frame = skip_initial_frames(game)
			cur_state = prepare_input(cur_frame)
		else:
			cur_state = prepare_input(next_frame, cur_state)

	out.release()

def skip_initial_frames(game):
	"""
	This function will skip some initial frames so that the ball and the opponent paddle comes in the view.

	:param game: pong game object
	:rtype: processed frame which will be used as the starting frame to train our DQN
	"""

	game.reset()
	next_frame = None
	for i in range(20):
		next_frame, reward = game.play(0)

	return next_frame

def cal_correct_q_values(rewards, v_values, are_terminals, gamma):
	correct_q_values = []

	for i in range(len(rewards)-1):
		if are_terminals[i]:
			correct_q_values.append(rewards[i])
		else:
			correct_q_values.append(rewards[i] + gamma * v_values[i+1])

	return array(correct_q_values, dtype=float32)
