# dqn.py 
This file is a tensorflow implementation of double DQN which has been trained for just 5000 episodes. If you want your model to be really good then you should train it for more episodes and can even try playing with the various hyper-parameters.

Model weights can be found at (https://drive.google.com/drive/folders/1VjJ9qp9jTMS3c9Yhr8oZR-AJmB-D4eYJ?usp=sharing)

* Episodes : 5000
* Max Steps of the game: 100
* Model tuning iterations: 100
* Batch size: 32
* Convolutional layers : 2
* Pooling layers: 1
* Learning Rate : 0.0001
* Initial epsilon: 1
* Epsilon-decay: 0.001
* Discount factor: 0.9

# policy_network.py 
This file is a tensorflow implementation of Policy Gradient which has been trained for just 90000 episodes. If you want your model to be really good then you should train it for more episodes and can even try playing with the various hyper-parameters.

Model weights can be found at (https://drive.google.com/open?id=1_mi-nVgFex0p_HQL7TAJlovTQD5lHsKj)

* Episodes : 90000
* Max Steps of the game: 256, 512, 1024, 2048
* Batch size: 256, 512, 1024, 2048
* Convolutional layers : 2
* Pooling layers: 1
* Learning Rate : 0.0001
* Discount factor: 0.9