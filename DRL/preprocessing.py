import numpy as np
import cv2

def preprocess_frame(img):
	"""
	This function will preprocess the input frame for DQN
	"""

	gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	crop = gray_scale[:, :]