import numpy as np
import data
from data import *

def logreg_train(X,Y_):
	"""
	Method for multiclass logistic regression training 
   
    Arguments
      X:  training data, np.array NxD
      Y_: class indices, np.array Nx1

    Return values
      w, b: logistic regression parameters
  """
	N = X.shape[0]
	C = int(max(Y_)+1)
	W = np.random.randn(X.shape[1],C)    #DxC
	b = np.zeros((1,C))                  #1xC
	param_iter = 1000
	param_delta = 0.1
	
	for i in range(param_iter):
		#exponential classification results
		scores = np.dot(X,W)+b                       #NxC
		expscores = np.exp(scores-np.amax(scores,axis=1,keepdims=True))    #NxC		
		
		#softmax denominator
		sumexp = np.vstack(np.sum(expscores,axis=1)) #Nx1

		#log values of class probabilities
		probs = expscores/sumexp.reshape(-1,1)       #NXC
		class_probs = probs[range(N),Y_[range(N),0]] #1xN
		log_class_probs = np.log(class_probs)        #1xN
	
		loss = (1.0/N)*np.sum(-log_class_probs)              #scalar

		# diagnostic print
		if i % 100 == 0:
			print("iteration {}: loss {}".format(i, loss))
	
		#derivations of loss components 
		dL_ds = probs
		dL_ds[range(N),Y_[range(N),0]] -= 1 #NxC
		
		#parameter gradients
		grad_W = np.transpose(np.dot(np.transpose(dL_ds),X))                    #DxC
		grad_b = np.transpose(np.sum(np.transpose(dL_ds), axis=1)[:,np.newaxis])#1xC
		#parameter updates
		W += -param_delta * grad_W
		b += -param_delta * grad_b
		
	return W,b
	
def logreg_classify(X,W,b):
	"""
	Method for calculating class probabilities
	
	Arguments
	X - training data, np.array NxD
	W - feature weights DxC
	b - bias 1xC
	
	Return Values
	probs - class probabilities for each data sample NxC
	"""
	scores = np.dot(X,W)+b #NxC
	expscores = np.exp(scores-np.max(scores)) #NxC
	sumexp = np.vstack(np.sum(expscores,axis=1)) #Nx1
	probs = expscores/sumexp
	return probs
	
def logreg_decfun(X,W,b):
	"""
	Decorator for  method logreg_classify
	"""
	def classify(X):
		return logreg_classify(X,W,b).argmax(axis=1)
	return classify
	
def sample_gauss_2d(C,N):
	"""
	Method for instantiating data
	
	Arguments
	C - number of classes 
	N - number of samples in each class
	
	Return Values
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

if __name__ =="__main__":
  np.random.seed(100)
  
  # get the training dataset
  X,Y_ = sample_gauss_2d(3,100)

  W,b = logreg_train(X,Y_)
  # evaluate the model on the training dataset
  probs = logreg_classify(X,W,b)
  print(probs.shape)
  # predicted classes
  Y = np.hstack([ np.argmax(probs[i][:]) for i in range(probs.shape[0])])
  
  # reshaping for other methods purposes
  Y_ = np.hstack(Y_)
  print(W)
  print(b)
  accuracy, pr, M= data.eval_perf_multi(Y, Y_)
  print("Accuracy: ",accuracy)
  print("Precision / Recall: ",pr)
  print("Confussion Matrix: ",M)
  
  # graph the decision surface
  decfun = logreg_decfun(X,W,b)
  bbox=(np.min(X, axis=0), np.max(X, axis=0))
  data.graph_surface(decfun, bbox, offset=0.5)

  # graph the data points
  data.graph_data(X, Y_, Y, special=[])

  # show the plot
  plt.show()
