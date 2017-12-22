import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import scipy.io
mat = scipy.io.loadmat('train_32x32.mat')

X = mat['X']
Y = mat['y']

X_train = np.zeros((X.shape[3] , X.shape[0] , X.shape[1] , X.shape[2]))

for i in range(10000):
    X_train[i,:,:,:] = X[:,:,:,i]
    

Y_train = np.zeros((10000 , 10))

for i in range(10000):
    for j in range(1,10):
        Y_train[i,j] = 0
    Y_train[i,Y[i]-1] = 1 

 
       
mat2 = scipy.io.loadmat('test_32x32.mat')

X2 = mat2['X']
Y2 = mat2['y']

X_test = np.zeros((X2.shape[3] , X2.shape[0] , X2.shape[1] , X2.shape[2]))

for i in range(1000):
    X_test[i,:,:,:] = X2[:,:,:,i]
    

Y_test = np.zeros((1000 , 10))

for i in range(1000):
    for j in range(1,10):
        Y_test[i,j] = 0
    Y_test[i,Y2[i]-1] = 1 

 
#plt.imshow(X[:,:,:,50])
#plt.show()
#print(Y_train[50])

print X_train[0,:,:,:]
   
def create_placeholders(n_h , n_w , n_c , n_y):

    X_train = tf.placeholder(shape=[None , n_h , n_w , n_c ] , dtype=tf.float32)
    Y = tf.placeholder(shape=[None , n_y] , dtype=tf.float32)
    
    return X, Y
    
def init_params():
    W1 = tf.get_variable("W1" , [5 , 5 , 3 , 18] , initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2" , [5 , 5 , 18 , 16] , initializer = tf.contrib.layers.xavier_initializer(seed=0))
    
    parameters = {"W1" : W1,
                  "W2" : W2
                  }
                  
    return parameters 
    
def forward_prop(X , parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    Z1 = tf.nn.conv2d(X , W1 , [1,2,2,1] , "SAME")
    A1 = tf.nn.relu(Z1)
    
    Z2 = tf.nn.conv2d(A1 , W2 , [1,2,2,1] , "SAME")
    A2 = tf.nn.relu(Z2)
    
    P2 = tf.contrib.layers.flatten(A2)

    Z3 = tf.contrib.layers.fully_connected(P2 , 100 , activation_fn = tf.nn.relu)

    Z4 = tf.contrib.layers.fully_connected(Z3 , 10 ,activation_fn= None)
    	
    return Z4
    
def compute_cost(Z4 , Y):
    print Z4.shape
    print Y.shape
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= (Z4) , labels = (Y))) 
    return cost  
    
def random_mini_batches(X , Y , mini_batch_size , seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation , :]
    shuffled_Y = Y[permutation , :]

    num_complete_minibatches = int(math.floor(m/mini_batch_size)) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size ,:]
        mini_batch_Y = shuffled_Y[k*mini_batch_size:(k+1)*mini_batch_size , :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[mini_batch_size*num_complete_minibatches:m , :]
        mini_batch_Y = shuffled_Y[mini_batch_size*num_complete_minibatches:m , :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches        
      
              

 
       
        
    

    
