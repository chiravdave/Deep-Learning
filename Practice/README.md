# mnist.py 
A DNN for image classification on mnist dataset. Below are the design details:
* Epochs : 25
* Conv-pool layers : 2
* Learning Rate : 0.001

# autoencoder.py
Autoencoder on mnist dataset. Below are the design details:
* Epochs : 80
* Encoder Hidden Layer Nodes : 50, 50, 28
* Decoder Hidden Layer Nodes : 28, 50, 50, 784
* Learning Rate : 0.001

# denoise_autoencoder.py
Autoencoder on mnist dataset. Below are the design details:
* Epochs : 20
* Encoder Hidden Layer Nodes : 250, 150, 100
* Decoder Hidden Layer Nodes : 100, 150, 250, 784
* Learning Rate : 0.001
