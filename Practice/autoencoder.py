import tensorflow as tf

def weight_initialize(shape):
  # Normal initialization with mean=0 and std=1
  weight = tf.Variable(tf.truncated_normal(shape))
  return weight

def bias_initialize(neurons):
  bias = tf.Variable(tf.zeros(neurons))
  return bias

def fc_layer(prev, input_size, output_size):
  weight = weight_initialize([input_size, output_size])
  bias = bias_initialize(output_size)
  return tf.nn.sigmoid(tf.matmul(prev, weight) + bias)

def auto_encoder(x):
  #Flatten input images
  flatten = tf.reshape(x, shape=[-1, 784])
  #Hidden layer 1 in the encoder
  enc_h1 = fc_layer(flatten, 784, 50)
  #Hidden layer 2 in the encoder
  enc_h2 = fc_layer(enc_h1, 50,50)
  #Latent layer
  lt = fc_layer(enc_h2, 50, 28)
  #Hidden layer 1 in the decoder
  dec_h1 = fc_layer(lt, 28, 50)
  #Hidden layer 2 in the decoder
  dec_h2 = fc_layer(dec_h1, 50, 50)
  #Output layer
  prediction = fc_layer(dec_h2, 50, 784)
  #Calculate batch loss
  loss = tf.reduce_mean(tf.squared_difference(flatten, prediction))
  return prediction, loss

def main():
  #Load mnist dataset
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  #Normalize images
  x_train, x_test = x_train / 255.0, x_test / 255.0
  #Create placeholders for the model
  x_input = tf.placeholder(tf.float32, shape=[None, 28, 28])
  #Build model
  prediction, loss = auto_encoder(x_input)
  #Minimize loss using an optimizer
  optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)
  with tf.Session() as sess:
    #Initialize all variables
    sess.run(tf.global_variables_initializer())
    #Start training
    for epoch in range(80):
      #Batch size of 32 will need 1875 passes to complete one epoch
      j = 0
      for i in range(1875):
        train_loss,_ = sess.run((loss, optimizer), feed_dict={x_input:x_train[j:(j+32)]})
        j = j + 32
      #Print loss at every 5th epochs
      if((epoch+1)%5==0):
        print('Loss at {} epoch is: {}'.format((epoch+1)/5,train_loss))
  
if __name__ == "__main__":
  main()
