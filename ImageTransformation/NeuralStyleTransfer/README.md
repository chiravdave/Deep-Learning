# Neural Style Transfer
This repository includes a tensorflow implementation of NST. NST is a method of generating an image which has content (objects and other high level features) of one image and style (texture and other low level features) of another image.

# Hyper Parameters
* Pre-trained model : VGG19
* Learning rate: 0.001
* Content layer weight: 1e4 (variable)
* Style layer weight: 1e-3 (variable)

# Additional Information
Different content images meld with different styles. One has to pick the right combination to generate an amazing image. Layer weights are variable. Keeping a right balance of the weight values for the layers can win the trick.

# Results
![Results](./style_transfer.gif)