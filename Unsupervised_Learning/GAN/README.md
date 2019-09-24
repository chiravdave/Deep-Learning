# GAN
This repository includes a tensorflow implementation of GAN on mnist dataset and it's been only trained for 1000 epochs. If you want your model to be really good then you should run it for more epochs.

* Epochs : 1000
* Learning rate: 0.0003
* Batch size: 128
* Learning Rate : 0.001

* Discriminator Network
	* Input layer: 784 neurons
	* FC layer 1: 256 neurons
	* Output layer: 1 neuron

* Generator Network
	* Input layer: 128 neurons
	* FC layer 1: 256 neurons
	* Output layer: 784 neurons