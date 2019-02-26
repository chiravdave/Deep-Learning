# Table of Contents
1. [Dependencies](README.md#dependencies)
1. [Problem](README.md#problem)
1. [Instructions to run the code](README.md#instructions-to-run-the-code)

# Dependencies
* python3
* numpy
* cv2
* tensorflow for GPU

# Problem
The problem is to create a visual object detection system for a customer to find the location of a phone dropped on 
the floor using a single RGB camera image. The customer has only one type of phone he is interested in detecting. Here is
an example of a image with a phone on it: <center> <img src="0.jpg"> </center>

# Instructions to run the code
Step 1. Install python3 if you don't have it. (https://realpython.com/installing-python/)

Step 2. Install argparse module if you don't have it. Run this command **pip3 install argparse** for installing it.

Step 3. Open the bash file **run.sh**, uncomment the last line and change python to python3. Once done, it should look like:
python3 ./src/pharmacy_counting.py ./input/itcont.txt ./output/top_cost_drug.txt
