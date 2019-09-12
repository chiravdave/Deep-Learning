import numpy as np
import cv2

def show_img(img):
	cv2.imshow('img', img)
	cv2.waitKey()
	cv2.destroyWindow('img')

def preprocess_frame(img):
	"""
	This function will preprocess the input frame for DQN
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
	# Flattening out the image
	flatten = crop.astype(np.float).ravel().reshape(1, 6400) 
	return flatten