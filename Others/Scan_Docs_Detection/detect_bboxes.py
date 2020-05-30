import numpy as np
import cv2
from typing import List, Tuple


def _create_blobs(
    img: np.ndarray,
    k_size: int,
    stride: int,
    threshold: int,
    input_w: int,
    output_w: int,
) -> np.ndarray:
    """
    This is a private function which will create white blobs around the objects in the image. It will run a kernel with
    the given stride over the input image and the generated output image, take weighted average of the pixel intensity
    and will assign white/black color to all the pixels at that particular patch on the output image based on the given
    threshold value.

    :param img: gray scale image with edges
    :param k_size: kernel/window size
    :param stride: stride along the axises
    :param threshold: threshold value to decide if all the pixels should be colored white/black
    :param input_w: weightage to the input image pixels
    :param output_w: weightage to the output image pixels
    :return: image with white blobs around the objects
    """

    height, width = img.shape[:2]
    """
    For storing output image with white colored blobs around the documents. Initially the image 
    is black without any identified documents.
    """
    blob_img = np.full((height, width), 0, dtype=img.dtype)

    for row in range(0, height - k_size, stride):
        for col in range(0, width - k_size, stride):
            average_intensity = np.average(
                img[row : row + k_size, col : col + k_size] * input_w
                + blob_img[row : row + k_size, col : col + k_size] * output_w
            )
            """
            If the average intensity is higher than the given threshold, consider that window on the input image as a 
            portion of a document
            """
            if average_intensity > threshold:
                blob_img[row : row + k_size, col : col + k_size] = 255

    return blob_img


def _draw_boxes(**kwargs) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    This is a private function which will try to approximate contours around the white blobs to rectangular boxes.

    :return: image with bounding boxes around the objects and a list of bounding box coordinates for every detected
             object in the form (x_min, y_min, x_max, y_max).
    """

    # Applying contour detection on the input image containing white blobs
    contours, hierarchy = cv2.findContours(
        kwargs["blob_img"], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    img = kwargs["img"]
    # List to store bounding box coordinates
    boxes = []
    for contour in contours:
        # Approximating contour shapes
        approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        # Considering only those shapes with vertices more than 4 and whose area is greater then a fixed threshold value
        if len(approx) >= 4 and area > 10000:
            # Getting a rectangular box around the contour
            rect = cv2.minAreaRect(contour)
            # Getting four corners of the box
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # Extracting coordinate values for the top left corner
            x_min, y_min = tuple(map(min, zip(*box)))
            # Extracting coordinate values for the bottom right corner
            x_max, y_max = tuple(map(max, zip(*box)))
            boxes.append((x_min, y_min, x_max, y_max))
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    return img, boxes


def detect_bbox(
    img_path: str, k_size=10, stride=3, threshold=80, input_w=2 / 3, output_w=1 / 3
) -> np.ndarray:
    """
    This function will return an image with bounding boxes around the objects in the input image. Only this function is
    to be used for external use.

    :param img_path: relative/absolute path of the image
    :param k_size: size of the kernel that will convolve over the input image to detect objects
    :param stride: value for the stride that will be used by the kernel
    :param threshold: threshold value to decide if all the pixels should be colored white/black
    :param input_w: weightage to the input image pixels
    :param output_w: weightage to the output image pixels
    :return: image with bounding boxes around the objects and a list of bounding box coordinates for every detected
             object in the form (x_min, y_min, x_max, y_max).
    """

    # Reading input image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # Converting into gray scale image
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blurring will help in reducing noise and making edges sharp
    blur = cv2.bilateralFilter(gray_scale, 5, 10, 50)
    # Detecting edges
    edges = cv2.Canny(blur, 50, 150, 3)
    # Performing dilation to increase the thickness of edges
    dilation_kernel = np.ones((5, 5), dtype=np.uint8)
    dilation = cv2.dilate(edges, dilation_kernel, iterations=1)

    # Generating white blobs around the documents
    blob_img = _create_blobs(dilation, k_size, stride, threshold, input_w, output_w)

    # Drawing bounding boxes around the documents
    bbox_img, bboxes = _draw_boxes(img=img, blob_img=blob_img)

    return bbox_img, bboxes
