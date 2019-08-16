import numpy as np
import cv2
import os

from glob import glob
from typing import Optional


def start_data_augmentation(
    directory_path: str, width: Optional[int] = None, height: Optional[int] = None
) -> None:
    """
    This function will perform various data augmentation methods like flipping, rotation and translation.

    :param directory_path: directory where the original images are
    :param width: width for our augmented images
    :param height: height for our augmented images
    :return: None
    """

    # Will store all the new images to the augmented_images folder
    if not os.path.isdir("../augmented_images"):
        os.mkdir("../augmented_images")
    # Looping through documents inside the directory
    for img_file in glob("{}/*.jpg".format(directory_path)):
        # Reading one document at a time
        img = cv2.imread(img_file)
        # Document name with extension
        img_extension = img_file.split("/")[-1]
        # Extracting document name
        img_name = img_extension[: img_extension.find(".jpg")]
        # Checking if output size is provided or not
        if width and height:
            img = cv2.resize(img, (width, height))
        flip_image(img, img_name)
        rotate_image(img, img_name)
        translate_image(img, img_name, 120)
        break


def flip_image(img: np.ndarray, img_name: str) -> None:
    """
    This function will flip the image left-right and up-down.

    :param img: original image
    :param img_name: name of the image file
    :return: None
    """

    # Flipping image left-right
    lr_flipped_img = np.fliplr(img)
    # Flipping image up-down
    ud_flipped_img = np.flipud(img)
    # Saving flipped images inside the directory
    cv2.imwrite("../augmented_images/{}_lrflip.jpg".format(img_name), lr_flipped_img)
    cv2.imwrite("../augmented_images/{}_udflip.jpg".format(img_name), ud_flipped_img)


def perform_rotation(img: np.ndarray, degree: int) -> np.ndarray:
    """
    This function will perform rotation on a given input image. It will make sure the object is not cut off and the
    object should be at the center of the image.
    
    :param img: input image
    :param degree: rotation angle (+ve = anti-clockwise and -ve = clockwise)
    :return: rotated image
    """

    # Getting height and width of the input image
    old_H, old_W = img.shape[:2]
    # Calculating center of the image
    center_X, center_Y = old_W / 2, old_H / 2

    # Getting rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((center_X, center_Y), degree, 1)
    # Getting cosine of the rotation
    cos_theta = abs(rotation_matrix[0, 0])
    # Getting sin of the rotation
    sin_theta = abs(rotation_matrix[0, 1])

    # Calculating new image height
    new_H = int((old_H * cos_theta) + (old_W * sin_theta))
    # Calculating new image width
    new_W = int((old_H * sin_theta) + (old_W * cos_theta))
    # Calculating translation required (using center coordinates)
    rotation_matrix[0, 2] += (new_W / 2) - center_X
    rotation_matrix[1, 2] += (new_H / 2) - center_Y

    # Performing rotation
    rotated_img = cv2.warpAffine(img, rotation_matrix, (new_W, new_H))
    return rotated_img


def rotate_image(img: np.ndarray, img_name: str) -> None:
    """
    This function will perform 90, 180 and 270 degree rotations.

    :param img: original image
    :param img_name: name of the image file
    :return: None
    """

    # Rotating copy of the image by 90, 180, 270 degree CCW
    for degree in range(90, 271, 90):
        copy_img = np.copy(img)
        cv2.imwrite(
            "../augmented_images/{}_rot{}.jpg".format(img_name, degree),
            perform_rotation(copy_img, degree),
        )


def translate_image(img: np.ndarray, img_name: str, shift_value: int) -> None:
    """
    This function will translate the image by the shift value along all the directions.

    :param img: original image
    :param img_name: name of the image file
    :param shift_value: pixels to translate the input image
    :return: None
    """

    copy_img = np.copy(img)
    rows, cols = img.shape[:2]
    # Translating image along the positive x-direction
    for row in range(rows - 1, -1, -1):
        for col in range(cols - 1, -1, -1):
            if col - shift_value >= 0:
                copy_img[row][col] = copy_img[row][col - shift_value]
            else:
                copy_img[row][col] = [255, 255, 255]
    cv2.imwrite("../augmented_images/{}_tranR.jpg".format(img_name), copy_img)

    # Translating image along the negative x-direction
    copy_img = np.copy(img)
    for row in range(rows):
        for col in range(cols):
            if cols - col > shift_value:
                copy_img[row][col] = copy_img[row][col + shift_value]
            else:
                copy_img[row][col] = [255, 255, 255]
    cv2.imwrite("../augmented_images/{}_tranL.jpg".format(img_name), copy_img)

    # Translating image along positive y-direction
    copy_img = np.copy(img)
    for row in range(rows - 1, -1, -1):
        for col in range(cols):
            if row - shift_value >= 0:
                copy_img[row][col] = copy_img[row - shift_value][col]
            else:
                copy_img[row][col] = [255, 255, 255]
    cv2.imwrite("../augmented_images/{}_tranD.jpg".format(img_name), copy_img)

    # Translating image along negative y-direction
    copy_img = np.copy(img)
    for row in range(rows):
        for col in range(cols):
            if rows - row > shift_value:
                copy_img[row][col] = copy_img[row + shift_value][col]
            else:
                copy_img[row][col] = [255, 255, 255]
    cv2.imwrite("../augmented_images/{}_tranU.jpg".format(img_name), copy_img)


if __name__ == "__main__":
    start_data_augmentation("tests/images")
