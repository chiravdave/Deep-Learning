import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob

def start_data_augmentation(directory_path, img_size):
    # Will store all the new images to the augmented_images folder
    if not os.path.isdir('../augmented_images'):
        os.mkdir('../augmented_images')
    # Looping through documents inside the directory
    for document in glob('{}/*.jpg'.format(directory_path)):
        # Reading one document at a time
        img = cv2.imread(document)
        # Document name with extension
        document_name_extension = document.split('/')[-1]
        # Extracting document name
        document_name = document_name_extension[:document_name_extension.find('.jpg')]
        if img.shape != img_size:
            img = cv2.resize(img, (img_size[0], img_size[1]))
        flip_image(img, document_name)
        rotate_image(img, document_name)
        translate_image(img, document_name, 120)
        cv2.imwrite('../augmented_images/{}.jpg'.format(document_name), img)

def flip_image(img, document_name):
    # Flipping image left-right
    lr_flipped_img = np.fliplr(img)
    # Flipping image up-down
    ud_flipped_img = np.flipud(img)
    # Saving flipped images inside the directory
    cv2.imwrite('../augmented_images/{}_lrflip.jpg'.format(document_name), lr_flipped_img)
    cv2.imwrite('../augmented_images/{}_udflip.jpg'.format(document_name), ud_flipped_img)

def rotate_image(img, document_name):
    # Rotating copy of the image by 90, 180, 270 degree CCW
    copy_img = np.copy(img)
    for i in range(1,4):
        copy_img = np.rot90(copy_img)
        cv2.imwrite('../augmented_images/{}_rot{}.jpg'.format(document_name, 90*i), copy_img)

def translate_image(img, document_name, shift_value):
    copy_img = np.copy(img)
    rows, cols = img.shape[:2]
    # Translating image along the right hand side
    for row in range(rows-1, -1, -1):
        for col in range(cols-1, -1, -1):
            if col-shift_value >= 0:
                copy_img[row][col] = copy_img[row][col-shift_value]
            else:
                copy_img[row][col] = [255, 255, 255]
    cv2.imwrite('../augmented_images/{}_tranR.jpg'.format(document_name), copy_img)
    copy_img = np.copy(img)
    # Translating image along the left hand side
    for row in range(rows):
        for col in range(cols):
            if cols-col > shift_value:
                copy_img[row][col] = copy_img[row][col + shift_value]
            else:
                copy_img[row][col] = [255, 255, 255]
    cv2.imwrite('../augmented_images/{}_tranL.jpg'.format(document_name), copy_img)
    copy_img = np.copy(img)
    # Translating image downwards
    for row in range(rows-1, -1, -1):
        for col in range(cols):
            if row-shift_value >= 0:
                copy_img[row][col] = copy_img[row-shift_value][col]
            else:
                copy_img[row][col] = [255, 255, 255]
    cv2.imwrite('../augmented_images/{}_tranD.jpg'.format(document_name), copy_img)
    copy_img = np.copy(img)
    # Translating image upwards
    for row in range(rows):
        for col in range(cols):
            if rows-row > shift_value:
                copy_img[row][col] = copy_img[row+shift_value][col]
            else:
                copy_img[row][col] = [255, 255, 255]
    cv2.imwrite('../augmented_images/{}_tranU.jpg'.format(document_name), copy_img)

if __name__ == '__main__':
    start_data_augmentation('../documents', (448, 448))