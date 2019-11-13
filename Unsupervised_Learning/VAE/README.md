# VAE
This repository includes a tensorflow implementation of VAE on mnist dataset with the netwrok been trained only for 4000 epochs. If you want your model to be really good then you should go for a bigger network and run it for more epochs. Model weights can be found at (https://drive.google.com/open?id=1Nis3NerxTnh1pjHqhxe0qvTyx39kOClq)

* Epochs : 4000
* Learning rate: 0.0003
* Batch size: 128
* Learning Rate : 0.001

* Encoder Network
	* Input layer: 784 neurons
	* FC layer 1: 256 neurons
	* Output layer: 10 neuron (each for mean and standard deviation)

* Decoder Network
	* Input layer: 10 neurons
	* FC layer 1: 256 neurons
	* Output layer: 784 neurons