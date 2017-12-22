import csv
import numpy as np
import math

filename = "mnist_train.csv"

fields = []
rows = []

with open(filename , 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    
    for row in csvreader:
        rows.append(row)
        
X_train = np.zeros((784 , 40000))
Y_temp_train = np.zeros(40000)
Y_train= np.zeros((10 , 40000))


X_cross = np.zeros((784 , 10000))
Y_temp_cross = np.zeros(10000)
Y_cross= np.zeros((10 , 10000))

X_test = np.zeros((784 , 10000))
Y_temp_test = np.zeros(10000)
Y_test= np.zeros((10 , 10000))


i = 0
for row in rows[:40000]:
    Y_temp_train[i] = float(row[0])
    j =0    
    for col in row[1:]:
        X_train[j,i] = float(col) 
        j+=1   
    i+=1
    
 
for i in range(Y_train.shape[1]):
    for j in range(10):
        Y_train[j,i] = 0
    Y_train[int(Y_temp_train[i]) , i] = 1

i = 0
for row in rows[40001:50000]:
    Y_temp_cross[i] = float(row[0])
    j =0    
    for col in row[1:]:
        X_cross[j,i] = float(col) 
        j+=1   
    i+=1
    
 
for i in range(Y_cross.shape[1]):
    for j in range(10):
        Y_cross[j,i] = 0
    Y_cross[int(Y_temp_cross[i]) , i] = 1


i = 0
for row in rows[50001:60000]:
    Y_temp_test[i] = float(row[0])
    j =0    
    for col in row[1:]:
        X_test[j,i] = float(col) 
        j+=1   
    i+=1
    
 
for i in range(Y_test.shape[1]):
    for j in range(10):
        Y_test[j,i] = 0



def softmax(Z):
    cache = (Z)
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    
    return A , cache
  
def relu(Z):
    A = Z * (Z>0)
    cache = (Z)
    return A , cache
    
def relu_backward(dA , cache):
   Z = cache
   dZ = dA * (Z>0)
   return dZ
    
    
def initialise_params(layer_dims):
    parameters = {}
    L = len(layer_dims)  
    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i] , layer_dims[i-1])*2/layers_dims[i-1]
        parameters['b' + str(i)] = np.zeros((layer_dims[i] , 1))
        
    return parameters
    
def linear_forward(A , W , b):
    Z = np.dot(W , A) + b
    cache = (A,W,b)
    return Z , cache
    
def linear_activation_forward(A_prev , W , b , lastlayer):
    Z , linear_cache = linear_forward(A_prev , W , b)
    
    if lastlayer==0:
        A , activation_cache = relu(Z)
    else:
        A , activation_cache = softmax(Z)    

    cache = (linear_cache , activation_cache)
    
    return A , cache 
    
def L_model_forward(X , parameters):
    caches = []
    A = X
    L = len(parameters)//2    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev , parameters['W' + str(l)] , parameters['b' + str(l)] , 0)
        caches.append(cache)
    AL, cache = linear_activation_forward(A , parameters['W' + str(L)] , parameters['b' + str(L)] , 1)
    caches.append(cache)   
    return AL , caches 
    
def compute_cost(AL , Y):
    m = Y.shape[1]
    cost = -1.0/m * np.sum(np.multiply(Y , np.log(AL)))
   
    return cost  
    
def linear_backward(dZ , cache):
    A_prev , W , b = cache
    m = A_prev.shape[1]
    
    dW = 1.0/m * np.dot(dZ , cache[0].T)
    db = 1.0/m * np.sum(dZ , axis=1 , keepdims=True)
    dA_prev = np.dot(cache[1].T , dZ)
    
    return dA_prev, dW, db
    
def linear_activation_backward(dA , cache):

    linear_cache, activation_cache = cache
    dZ = relu_backward(dA , activation_cache)
    dA_prev, dW, db = linear_backward(dZ,linear_cache) 
    
    return dA_prev , dW , db   
    
def L_model_backward(AL , Y , caches):
    grads={}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
                  
    dZ = AL - Y
    current_cache = linear_backward(dZ, caches[L-1][0])
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = current_cache
    
    for l in reversed(range(L-1)):
        current_cache = linear_activation_backward(grads["dA"  + str(l+2)] , caches[l])
        dA_prev_temp, dW_temp, db_temp = current_cache
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def random_mini_batch(X , Y , mini_batch_size):
    np.random.seed(0)
    m = X.shape[1]
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    num_complete_minibatches = int(math.floor(m/mini_batch_size)) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,mini_batch_size*num_complete_minibatches:m]
        mini_batch_Y = shuffled_Y[:,mini_batch_size*num_complete_minibatches:m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches        
         
             
def update_params(parameters , grads , alpha):
    L = len(parameters)/2 
    
    for l in range(1 , L):
        parameters["W" + str(l)]-= alpha*grads["dW" + str(l)]
        parameters["b" + str(l)]-= alpha*grads["db" + str(l)]
    return parameters    
    
    
def L_layer_model(X , Y, layer_dims , alpha = 0.01 , iterations = 500):
    costs = []
    
    parameters = initialise_params(layer_dims)
    for i in range(0, iterations):
        batches = random_mini_batch(X , Y , 256)
        for mini_batch in batches:
            mini_batch_X , mini_batch_Y = mini_batch
            AL, caches = L_model_forward(mini_batch_X , parameters)
            cost = compute_cost(AL , mini_batch_Y)
            grads = L_model_backward(AL, mini_batch_Y , caches)
            parameters = update_params(parameters , grads , alpha)
        
            if i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if  i % 100 == 0:
                costs.append(cost) 
            
    return parameters

def predict(X_test , Y_test , parameters):
    AL , cache = L_model_forward(X_test , parameters)
    AL=AL.T
    Y_test = Y_test.T
    correct_preds = 0.0
    for i in range(AL.shape[0]):
        if(np.where(AL[i,:]==AL[i,:].max())==(np.where(Y_test[i]==1)))==True:
			correct_preds+=1.0
	print float(correct_preds)/float(Y_test.shape[0])
	
    
layers_dims = [784 , 100 , 10]    
parameters = L_layer_model(X_train, Y_train, layers_dims, alpha = 0.01 , iterations = 100) 
predict(X_cross , Y_cross , parameters)
predict(X_test , Y_test , parameters)


                        


           
