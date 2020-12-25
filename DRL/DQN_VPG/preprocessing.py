from numpy import reshape, append, concatenate, float32, ndarray
import cv2

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
	crop = crop.astype(float32)
	# Normalizing the image
	crop = crop / 255
	crop = reshape(crop, (80, 80, 1)) 
	return crop

def prepare_input(cur_frame, prev_state = None):
	processed_frame = preprocess_frame(cur_frame)
	if isinstance(prev_state, ndarray):
		return append(processed_frame, prev_state[:, :, 0:3], axis=-1)
	else:
		# Stacking 4 frames to capture motion
		return concatenate((processed_frame, processed_frame, processed_frame, processed_frame), axis=-1)