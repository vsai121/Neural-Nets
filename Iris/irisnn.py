import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
Ytemp = iris.target

Y = np.zeros((X.shape[0] , 3))

for i in range(X.shape[0]):
    if Ytemp[i]==0:
        Y[i,0] = 1
        Y[i,1] = 0
        Y[i,2] = 0
    if Ytemp[i]==1:
        Y[i,0] = 0
        Y[i,1] = 1
        Y[i,2] = 0
    if Ytemp[i]==2:
        Y[i,0] = 0
        Y[i,1] = 0
        Y[i,2] = 1

             
def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)

    return A         

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    
def sigmoid_grad(z):
    return np.multiply(sigmoid(z) * (1-sigmoid(z)))    
    
def neural_net(X , Y):
    n_x = X.shape[1]
    n_h = 4
    n_y = 3
    
    return (n_x , n_h , n_y)
    
def initialize_params(n_x , n_h , n_y):
    W1 =  np.random.randn(n_h , n_x) * 0.01
    b1 =  np.zeros((n_h , 1))
    W2 =  np.random.randn(n_y , n_x) * 0.01
    b2 =  np.zeros((n_y , 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
    

def forward_prop(X , parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1 , X.T)  + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2 , A1) + b2
    A2 = softmax(Z2)
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache
    

def compute_cost(A2 , Y , parameters):

    m = X.shape[0]
    logCost = np.multiply(Y.T , np.log(A2))
    cost = -1.0/m * np.sum(logCost)  
    
    return cost
    

def back_prop(parameters , cache , X , Y):
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]
    
    m = X.shape[0]
    
    dZ2 = A2 - Y.T
    dW2 = 1.0/m * np.dot(dZ2 , A1.T)
    db2 = 1.0/m * np.sum(dZ2 , axis=1 , keepdims = True)
    
    dZ1 = np.dot(W2.T , dZ2) * A1 * (1-A1)
    dW1 = 1.0/m * np.dot(dZ1 , X)
    db1 = 1.0/m * np.sum(dZ1 , axis=1 , keepdims = True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads
    
def gradient_descent(parameters , grads , alpha = 0.01):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
  
   
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
   
    W1 = W1 - alpha*dW1
    W2 = W2 - alpha*dW2
    b1 = b1 - alpha*db1
    b2 = b2 - alpha*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def nn_model(X , Y , n_h , iterations = 15000):
    np.random.seed(3)
    n_x = X.shape[1]
    n_y = Y.shape[1]
    
    parameters = initialize_params(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(0, iterations):
        cost = 0
        A2, cache = forward_prop(X , parameters)
        cost = cost +  compute_cost(A2 , Y , parameters)
        grads = back_prop(parameters , cache , X , Y)
        parameters = gradient_descent(parameters, grads)
        
        if i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters  

def predict( X , Y):
    parameters = nn_model(X , Y , 4)
    A2 , cache = forward_prop(X , parameters)  
    prediction = A2.T
    print prediction==prediction.max(axis=1 , keepdims=True)   
 
predict(X,Y)
    
    
            
    
            
