import numpy as np
import cv2

from glob import glob
from typing import Optional, Any
from os import mkdir
from os.path import isdir, splitext


def augment_images(
    src_dir: str, new_config: Dict[str, Any]
) -> None:
    """
    This function will perform various data augmentation methods like flipping, rotation, translation
    and brightness adjustments

    :param src_dir: absolute path of the directory containing original images
    :param new_config: configuration to use during data augmentation
    :return: None
    """

    # Will store all the new images under a folder names as augmented_images
    if not isdir("augmented_images"):
        mkdir("augmented_images")

    config = {
    "lr_flip": True, "ud_flip": True, "pixel_shifts": [4, 10], "rotation_angles": [10, -10], 
    "center_rotated_image": True, "bt_increase_th": 1.2, "bt_decrease_th": 0.7
    }
    config.update(new_config)
    
    # Looping through all the images inside the src directory
    for img_file in glob("{}/*".format(src_dir)):
        img = cv2.imread(img_file)
        if "float" not in img.dtype:
            img /= 255.0
        file_name_with_extension = splitext(img_file.split("/")[-1])
        file_name, file_extension = file_name_with_extension[0], file_name_with_extension[1]
        flip_image(img, file_name, file_extension, config)
        rotate_image(img, file_name, file_extension, config)
        translate_image(img, file_name, file_extension, config)
        adjust_brightness(img, file_name, file_extension, config)


def flip_image(img: np.ndarray, file_name: str, file_extension: str, config: Dict[str, Any]) -> None:
    """
    This function will flip the image left-right and up-down.

    :param img: original image
    :param file_name: name of the image file
    :param file_extension: file extension of the original image file
    :param config: configuration to use during data augmentation
    :return: None
    """

    # Checking for flipping image left-right
    if config["lr_flip"]:
        cv2.imwrite(f"augmented_images/{file_name}_lrflip{file_extension}", np.fliplr(img))

    # Checking for flipping image upside-down
    if config["ud_flip"]:
        cv2.imwrite(f"augmented_images/{file_name}_udflip{file_extension}", np.flipud(img))


def rotate_image(image: np.ndarray, file_name: str, file_extension: str, config: Dict[str, Any]) -> None:
    """
    This function will rotate the given image using the angles specified in the config.

    :param img: original image
    :param file_name: name of the image file
    :param file_extension: file extension of the original image file
    :param config: configuration to use during data augmentation
    :return: None
    """

    img_width, img_height = image.shape[1], image.shhape[0]
    for degree_angle in config["rotation_angles"]:
        rt_matrix = cv2.getRotationMatrix2D((img_width/2, img_height/2), degree_angle, 1)
        if config["center_rotated_image"]:
            # Getting sin & cosine of the rotation
            sin_theta, cos_theta = abs(rt_matrix[0, 1]), abs(rt_matrix[0, 0])
            # Calculating new image height
            new_H = int((old_H * cos_theta) + (old_W * sin_theta))
            # Calculating new image width
            new_W = int((old_H * sin_theta) + (old_W * cos_theta))
            # Calculating translation required to bring the image to the center
            rt_matrix[0, 2] += (new_W / 2) - img_width/2
            rt_matrix[1, 2] += (new_H / 2) - img_height/2
            rt_img = cv2.warpAffine(image, rt_matrix, (new_W, new_H), flags=cv2.INTER_CUBIC)
            rt_img = cv2.resize(rt_img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        else:
            rt_img = cv2.warpAffine(image, rt_matrix, (img_width, img_height), flags=cv2.INTER_CUBIC)
        
        cv2.imwrite(f"augmented_images/{file_name}_rot{degree_angle}{file_extension}", rt_img)


def translate_image(image: np.ndarray, file_name: str, file_extension: str, config: Dict[str, Any]) -> None:
    """
    This function will translate the given image along the diagonal directions.

    :param img: original image
    :param file_name: name of the image file
    :param file_extension: file extension of the original image file
    :param config: configuration to use during data augmentation
    :return: None
    """

    ts_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=image.dtype)
    for shift in config["pixel_shifts"]:
        for t_x, t_y in [(-shift,-shift), (-shift,shift), (shift,-shift), (shift,shift)]:
            ts_matrix[0][2], ts_matrix[1][2] = t_x, t_y
            cv2.imwrite(
                f"augmented_images/{file_name}_trans{shift}{file_extension}", 
                cv2.warpAffine(image, ts_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
                )

def adjust_brightness(image: np.ndarray, file_name: str, file_extension: str, config: Dict[str, Any]) -> None:
    """
    This function will change the brightness of the input image.

    :param img: original image
    :param file_name: name of the image file
    :param file_extension: file extension of the original image file
    :param config: configuration to use during data augmentation
    :return: None
    """

    # Increasing brightness
    cv2.imwrite(
            f"augmented_images/{file_name}_bt_inc{config['bt_increase_th']}{file_extension}", 
            np.clip(image*config['bt_increase_th'], 0.0, 1.0)
            )
    # Decreasing brightness
    cv2.imwrite(
        f"augmented_images/{file_name}_bt_inc{config['bt_decrease_th']}{file_extension}", 
        np.clip(image*config['bt_decrease_th'], 0.0, 1.0)
        )

augment_images("/Users/chirav/Documents/Deep-Learning/Utils")