# Table of Contents
1. [Introduction](README.md#introduction)
1. [Dependencies](README.md#dependencies)
1. [Problem](README.md#problem)
1. [Instructions to run the code](README.md#instructions-to-run-the-code)

#Introduction
Architected a Convolutional Neural Network to localize a phone in an image with a minimal amount of data to train on (training images = 130) Below are the design details:

* Epochs : 165
* Training Mode : Batch
* Hidden Layers : 2
* Learning Rate : 0.0001

# Dependencies
* python3
* numpy
* cv2
* sklearn
* tensorflow for GPU

# Problem
The problem is to create a visual object detection system for a customer to find the location of a phone dropped on 
the floor using a single RGB camera image. The customer has only one type of phone he is interested in detecting. Here is
an example of a image with a phone on it: <p align="center"> <img src="0.jpg"> </p>

# Instructions to run the code
Step 1. Install all the necessary packages

Step 2. Run the training script named train_phone_finder.py with image direcory passed as an argument.

Step 3. Once training is done run the testing script named find_phone.py with the image file that you want to test as an argument. 
