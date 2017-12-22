import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
Ytemp = iris.target

Y = np.zeros((Ytemp.shape[0] , 1))

for i in range(Ytemp.shape[0]):
  Y[i,0] = Ytemp[i]


def sigmoid(z):
 return 1.0 / (1.0 + np.exp(-z))
 
def initialise_weights(dim):
 w = np.zeros((dim,1))
 b = 0
 
 assert(w.shape == (dim, 1))
 return w , b


def propogate(w,b,X,Y):

 m = X.shape[0]
 Z = np.dot(X,w) + b
 A = sigmoid(Z)

 Cost = -1.0/m* np.sum(Y*np.log(A) + (1-Y)*(np.log(1-A)))
 
 dw = 1.0/m * (np.dot(X.T , (A-Y)))
 db = 1.0/m * np.sum(A-Y)
 
 
 grads = {"dw": dw,"db": db}
    
 return grads, Cost
 
 
 
def optimise(w , b , X , Y , alpha , iterations):

    costs=[]
    for i in range(iterations):
        grads , cost = propogate(w , b , X , Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - alpha*dw
        b = b - alpha*db
        
        if i % 100 == 0:
          costs.append(cost)
        
        if i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
  
    params = {"w": w,"b": b}
    
    grads = {"dw": dw,"db": db}
    
    return params, grads, costs   
    

    
def onevsall(w , b  , labels , X , Y , alpha , iterations):

    m = X.shape[0]
    Y_prediction = np.zeros((m,labels))
    
    
    for i in range(labels):
        params , grads , costs = optimise(w , b , X , (Y==i).astype(int) , alpha , iterations)
        w = params["w"]
        b = params["b"]

        for j in range(m):
            Y_prediction[j,i] = sigmoid(np.dot(X,w)+b)[j,0]
        
    print Y_prediction == Y_prediction.max(axis=1, keepdims=1)
      
w,b = initialise_weights(X.shape[1])

onevsall(w,b,3,X,Y,0.2,1500)


    
         
           
            
                
