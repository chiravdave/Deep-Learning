import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

noise_scale = 1

def weight_initialize(shape):
  # Normal initialization with mean=0 and std=1
  weight = tf.Variable(0.1*tf.random.truncated_normal(shape))
  return weight

def bias_initialize(neurons):
  bias = tf.Variable(tf.zeros(neurons))
  return bias

def fc_layer(prev, input_size, output_size):
  weight = weight_initialize([input_size, output_size])
  bias = bias_initialize(output_size)
  return tf.nn.sigmoid(tf.matmul(prev, weight) + bias)

def denoised_auto_encoder(x, x_noisy):
  #Flatten input images
  flatten = tf.reshape(x, shape=[-1, 784])
  flatten_noisy = tf.reshape(x_noisy, shape=[-1, 784])
  #Hidden layer 1 in the encoder
  enc_h1 = fc_layer(flatten_noisy, 784, 250)
  #Hidden layer 2 in the encoder
  enc_h2 = fc_layer(enc_h1, 250,150)
  #Latent layer
  lt = fc_layer(enc_h2, 150, 100)
  #Hidden layer 1 in the decoder
  dec_h1 = fc_layer(lt, 100, 150)
  #Hidden layer 2 in the decoder
  dec_h2 = fc_layer(dec_h1, 150, 250)
  #Output layer
  prediction = fc_layer(dec_h2, 250, 784)
  #Calculate batch loss
  loss = tf.reduce_mean(tf.math.squared_difference(flatten, prediction))
  return prediction, loss

def noisy_image(images):
  n_rows = images.shape[1]
  n_cols = images.shape[2]
  noise = noise_scale * np.random.normal(0, 0.1, size=(n_rows, n_cols))
  noisy_images = images + noise
  return noisy_images

def show_images(original, noise, generated):
  plt.subplot(1, 3, 1)
  plt.imshow(original, cmap="gray")
  plt.title('Original')
  plt.subplot(1, 3, 2)
  plt.imshow(noise, cmap="gray")
  plt.title('Distorted')
  plt.subplot(1, 3, 3)
  plt.imshow(generated, cmap="gray")
  plt.title('Generated')
  plt.show()

def main():
  # It is safe to clear the computation graph because there still could be variables present from the previous run  
  tf.compat.v1.reset_default_graph()
  #Load mnist dataset
  (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
  #Normalize Images
  x_train = x_train.astype(np.float32)/255
  x_test = x_test.astype(np.float32)/255
  #Create placeholders for the model
  x_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28])
  x_noisy_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28])
  #Build model
  prediction, loss = denoised_auto_encoder(x_input, x_noisy_input)
  #Minimize loss using an optimizer
  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
  with tf.compat.v1.Session() as sess:
    #Initialize all variables
    sess.run(tf.compat.v1.global_variables_initializer())
    #Start training
    for epoch in tqdm(range(30)):
      #Batch size of 32 will need 1875 passes to complete one epoch
      j = 0
      for i in range(1875):
        #Add noise to images
        x_noisy_train = noisy_image(x_train[j:(j+32)])
        train_loss,_ = sess.run((loss, optimizer), feed_dict={x_input:x_train[j:(j+32)], x_noisy_input:x_noisy_train})
        j = j + 32
      #Print loss at every 5th epochs
      if((epoch+1)%5==0):
        print('Loss at {} epoch is: {}'.format((epoch+1),train_loss))
    # For testing the model
    while(True):
      choice = raw_input('Do you want to continue? y/n').lower()
      if choice == 'y':
        k = int(input('Enter the image number you want to test'))
        test_image = x_test[k:k+1]
        # Multiply with 255 as data type of mnist is uint8
        test_noise = noisy_image(test_image)*255
        generated = sess.run((prediction), feed_dict={x_noisy_input: test_noise})
        show_images(x_test[k], np.reshape(test_noise, (28,28)), np.reshape(generated, (28,28)))
      else:
        break
        
if __name__ == "__main__":
  main()
