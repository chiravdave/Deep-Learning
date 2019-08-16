# Table of Contents
1. [Introduction](README.md#introduction)
1. [Dependencies](README.md#dependencies)
1. [Instructions to run the code](README.md#instructions-to-run-the-code)

# Introduction
The recent success of deep learning in computer vision can be partially attributed to the availability of large amounts of labeled data to train on. A model’s performance typically improves as you increase the quality, diversity and the amount of training data. However, collecting enough quality data to train a model to perform well is often prohibitively difficult. Hence, it is very important to have diverse data samples for your specific task. With this code, you will be able to generate new data samples that will have images resized and with objects flipped, rotated and translated.

# Dependencies
* python3
* numpy
* cv2

# Instructions to run the code
Step 1. Install all the necessary package dependencies.

Step 2. Once the project is downloaded, you just need to call the function <b>start_data_augmentation</b> with image directory and output image dimensions as width and height (optional) as arguments, i.e <b>start_data_augmentation('../documents', 448, 448)</b>. 