import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import math

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict
    
def convert_images(raw):
  
    # Reshape the array to 4-dimensions.
    images = raw.reshape([-1, 3, 32 ,32])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images
   
images = unpickle("./cifar-10-batches-py/data_batch_1")

X = images['data']
Y = images['labels']

X_train = convert_images(X)

Y_train= np.zeros((X.shape[0] , 10))

for i in range (len(Y)):
    for j in range(10):
        Y_train[i][j] = 0
    Y_train[i,Y[i]] = 1
    
images2 = unpickle("./cifar-10-batches-py/data_batch_2")

X2 = images['data']
Y2 = images['labels']

X_test = convert_images(X2)

Y_test= np.zeros((X2.shape[0] , 10))

for i in range (len(Y2)):
    for j in range(10):
        Y_test[i][j] = 0
    Y_test[i,Y2[i]] = 1   
    
     
#plt.imshow(X_train[25,:,:,:])
#plt.show()   

#print X[0,:]
#print X_train[0,:,:,:] 


#print Y_train[0:5,:]

def create_placeholders(n_h , n_w , n_c , n_y):
    X = tf.placeholder(shape=[None , n_h , n_w , n_c] , dtype=tf.float32)
    Y = tf.placeholder(shape=[None , n_y] , dtype=tf.float32)
    return X , Y

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  

def init_parameters():

    W1 = weight_variable([5,5,3,32])
    b1 = bias_variable([32])
    
    W2 = weight_variable([5,5,32,32])
    b2 = bias_variable([32])
      
    W3 = weight_variable([5,5,32,64])
    b3 = bias_variable([64])
    
    
    parameters ={'W1' : W1,
                 'b1' : b1,
                 'W2' : W2,
                 'b2' : b2,
                 'W3' : W3,
                 'b3' : b3}
                 
    return parameters
    

def forward_prop(X , parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    
    Z1 = conv2d(X , W1)
    A1 = tf.nn.relu(Z1 + b1)
    
    P1 = max_pool(A1)
    
    Z2 = conv2d(P1 , W2)
    A2 = tf.nn.relu(Z2 + b2)
    
    P2 = max_pool(A2)
    
    Z3 = conv2d(P2 , W3)
    A3 = tf.nn.relu(Z3 + b3)
    
    P3 = tf.contrib.layers.flatten(A3)
    Z4 = tf.contrib.layers.fully_connected(P3 , 64)
    
    Z5 = tf.contrib.layers.fully_connected(Z4 , 10 , activation_fn = None)
    print Z5.shape
    return Z5
    
def compute_cost(Z5 , Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= (Z5) , labels = (Y)))
    return cost
    
def random_mini_batches(X , Y , mini_batch_size , seed):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation , :]
    shuffled_Y = Y[permutation , :]

    num_complete_minibatches = int(math.floor(m/mini_batch_size)) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size ,:]
        mini_batch_Y = shuffled_Y[k*mini_batch_size:(k+1)*mini_batch_size ,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[mini_batch_size*num_complete_minibatches:m , :]
        mini_batch_Y = shuffled_Y[mini_batch_size*num_complete_minibatches:m , :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches 
    
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.001,
          num_epochs = 100, minibatch_size = 64, print_cost = True):
    
    ops.reset_default_graph()                         
    tf.set_random_seed(1)                             
    seed = 3                                        
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0 , n_W0 , n_C0 , n_y)

    # Initialize parameters
    parameters = init_parameters()

  
    # Forward propagation: Build the forward propagation in the tensorflow graph

    Z5 = forward_prop(X , parameters)

    
    # Cost function: Add cost function to tensorflow graph

    cost = compute_cost(Z5 , Y)

    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer , cost] , {X:minibatch_X , Y:minibatch_Y})
       
                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
       
        # Calculate the correct predictions
        predict_op = tf.argmax(Z5, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters



train_acc , test_acc , parameters = model(X_train, Y_train, X_test, Y_test)

           
           
         
    
        
    
    
    
                                
        
    
        
         

 




