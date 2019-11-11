import numpy as np
import cv2

from pong import Pong

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

	return preprocess_frame(next_frame)

def show_img(img):
	"""
	This function will help to visualize the input image.

	:param img: input image
	"""

	cv2.imshow('img', img)
	cv2.waitKey()
	cv2.destroyWindow('img')

def preprocess_frame(img):
	"""
	This function will preprocess the input image/frame for DQN.

	:param img: input image/frame
	"""

	# We don't need scores
	crop = img[35:195]
	# Downsampling by a factor of 2 and taking only the first channel
	crop = crop[::2, ::2, 0]
	# Erasing background
	crop[crop == 144] = 0
	crop[crop == 109] = 0
	# Only ball and paddle should be visible
	crop[crop != 0] = 255
	# Converting into float type
	crop = crop.astype(np.float)
	# Normalizing the image
	crop = crop / 255
	crop = np.reshape(crop, (80, 80, 1)) 
	return crop