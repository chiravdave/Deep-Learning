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
	# Converting into float type
	crop = crop.astype(np.float)
	# Normalizing the image
	crop = crop / 255
	crop = np.reshape(crop, (80, 80, 1)) 
	return crop