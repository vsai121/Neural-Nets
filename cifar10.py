import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
 
#plt.imshow(X_train[25,:,:,:])
#plt.show()   

#print X[0,:]
#print X_train[0,:,:,:] 


#print Y_train[0:5,:]

def create_placeholders(n_h , n_w , n_c , n_y):
    X = tf.placeholder(shape=[None , n_h , n_w , n_c] , dtype=tf.float32)
    Y = tf.placeholder(shape=[None , n_y] , dtype=tf.float32)
    Y_class = tf.argamax(Y , dimension=1)
    return X , Y , Y_class

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
    b1 = bias_variable([32])
      
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
    Z1 = conv2d(X , W1)
    A1 = tf.nn.relu(Z1 + b1)
    
    P1 = max_pool(A1)
    
    Z2 = conv2d(P1 , W2)
    A2 = tf.nn.relu(Z2 + b2)
    
    P2 = max_pool(A2)
    
    Z3 = conv2d(P2 , W3)
    A3 = tf.nn.relu(Z3 + b3)
    
    Z4 = tf.contrib.layers.fully_connected(Z3 , 64)
    
    Z5 = tf.contrib.layers.fully_connected(Z4 , 10 , activation_fn = None)
    
    return Z5
    
        
    
    
    
                                
        
    
        
         

 




