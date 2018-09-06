import numpy as np
import cv2

def normalizeImage(image):
    return ((image - image.min())/ (image.max() - image.min()))  #Normalizing Image

def generateData(images_dir, image_width, image_height, channels):
    image_paths = []  # store absolute path of all the images
    labels = []       #Storing center co-ordinates of the phone in every image
    labels_file = images_dir+'/labels.txt'     #Path of the labels.txt file
    with open(labels_file) as fp:
        for line in fp:
            line = line.rstrip('\n')
            split = line.split(' ')
            image_paths.append(images_dir+'/'+split[0])                #Storing image paths
            labels.append([float(split[1]), float(split[2])])          #Storing labels
    i = 0
    features = np.ndarray(shape=(len(image_paths), image_height, image_width, channels), dtype = np.float32)
    for image in image_paths:          #Reading Images one at a time
        bgr_image = cv2.imread(image)
        b, g, r = cv2.split(bgr_image) #get b,g,r values of the image
        rgb_image = cv2.merge([r,g,b]) #convert into r,g,b format
        normalized_image = normalizeImage(rgb_image) 
        features[i] = normalized_image
        i = i + 1
    labels_numpy_array = np.array(labels, dtype = np.float32) #Converting labels list into numpy array
    return features, labels_numpy_array
