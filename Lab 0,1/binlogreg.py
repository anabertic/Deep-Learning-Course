import numpy as np
import data
from data import *

def binlogreg_train(X,Y_):
  """
  Method for binary logistic regression training
  Arguments
      X:  training data, np.array NxD
      Y_: class indices, np.array Nx1

  Return values
      w, b: logistic regression parameters
  """
  N = X.shape[0] #number of samples
  D = X.shape[1] #number of features
  w = np.random.randn(D,1)
  b = 0
  
  param_niter = 5000
  param_delta = 0.001
  
  #gradient descent
  for i in range(param_niter):
	  #classification scores
	  scores = np.dot(X,w) + b #Nx1
	  
	  #probabilities of class c_1	 
	  probs = 1 / (1 + np.exp(-scores))    #Nx1
	  
	  #cross entropy loss
	  loss = np.sum(-Y_*np.log(probs)-(1-Y_)*np.log(1-probs))
	  
	  #diagnostic print
	  if i %10 ==0:
		  print("iteration {}: loss {}".format(i,loss))
	  
	  #loss derivation via classification scores
	  dL_scores = probs-Y_  #Nx1
	  
	  #parameter gradients
	  grad_w = (np.dot(dL_scores.T,X)).T#Dx1
	  grad_b = np.sum(dL_scores)        #scalar
	  
	  #updated parameters
	  w += -param_delta * grad_w
	  b += -param_delta * grad_b

  return w,b

def binlogreg_classify(X,w,b):
	"""
	Method for calculating class probabilities
	
	Arguments
	X - training data, np.array NxD
	w - feature weights DxC
	b - bias 1xC
	
	Return Values
	probs - class probabilities for each data sample NxC
	"""
	scores = np.dot(X,w) + b #Nx1
	probs = 1 / (1 + np.exp(-scores))
	return probs

def sample_gauss_2d(C,N):
	"""
	Method for instantiating data
	
	Arguments
	C - number of classes 
	N - number of samples in each class
	
	Return values
	x - array of features NxC
	Y - array of labels   Nx1
	"""
	x = []
	Y = []
	for i in range(0,C):
		G = data.Random2DGaussian()
		samples=G.get_sample(N)
		x.append(samples.tolist())
		Y.append([i for j in range(N)])
	x = np.reshape(x,(N*C,2))
	Y = np.reshape(Y,(N*C,1))
	return x,Y

def binlogreg_decfun(X,w,b):
    """
    Decorator for  method logreg_classify
    """
    def classify(X):
        return binlogreg_classify(X, w,b)
    return classify  
	
if __name__=="__main__":
  np.random.seed(100)

  # get the training dataset
  X,Y_ = sample_gauss_2d(2,100)

  # train the model
  w,b = binlogreg_train(X,Y_)

  # evaluate the model n the training dataset
  probs = binlogreg_classify(X,w,b)
  print(probs)
  #predicted labels
  Y = np.vstack([ 0 if (probs[i]<0.5) else 1 for i in range(Y_.shape[0])])

  #evaluation metrics
  accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
  print("Accuracy: ",accuracy)
  print("Precision: ",precision)
  print("Recall: ",recall)  
  AP = data.eval_AP(np.vstack(Y_[probs.reshape(1,200).argsort()]))
  print("Average precision: ",AP)
  
  #graph the decision surface
  decfun = binlogreg_decfun(X,w,b)
  bbox=(np.min(X, axis=0), np.max(X, axis=0))
  data.graph_surface(decfun, bbox, offset=0.5)

  # graph the data points
  data.graph_data(X, np.hstack(Y_), np.hstack(Y), special=[])

  # show the plot
  plt.show()



	  

  
  
  
