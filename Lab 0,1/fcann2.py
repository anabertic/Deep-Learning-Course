import numpy as np
import data
from data import *

def ReLU(x):
	return np.maximum(0,x)
	
def fcann_train(X,Y_,h,param_iter,param_delta,reg):
	"""
	Method for probabilistic classification model training 
	(Back propagation gradient descent)
	Arguments
		X:  training data, np.array NxD
		Y_: class indices, np.array Nx1
		h: size of hidden layer
		param_iter: number of iterations during training
		param_delta: learning rate
		reg: regularization coefficient
	Return values
		W1, b1, W2, b2: model parameters
	"""	
	N = X.shape[0]
	C = np.max(Y_)+1
	
	#weights initialization
	W1 = np.random.randn(X.shape[1],h) #Dxh
	b1 = np.zeros((1,h))               #1xh
	W2 = np.random.randn(h,C)          #hxC
	b2 = np.zeros((1,C))               #1xC
  
	for i in range(param_iter):
		# forward Pass - evaluating class scores
		h1 = ReLU(np.dot(X,W1)+b1)	#hidden layer input Nxh
		s2 = np.dot(h1,W2)+b2       #NxC
		
		# class probabilities
		exp_s2 = np.exp(s2-np.max(s2))               #NxC
		sumexp = np.vstack(np.sum(exp_s2,axis = 1))
		
		# log values of class probabilities
		probs = exp_s2/sumexp.reshape(-1,1)          #NXC softmax
		class_probs = probs[range(N),Y_]             #1xN
		log_class_probs = np.log(class_probs)        #1xN
	
		loss = np.sum(-log_class_probs)/N            #scalar
		
		# diagnostic print
		if i % 100 == 0:
			print("iteration {}: loss {}".format(i, loss))
		
		# derivations of loss components 
		dL_ds2 = (1/N)*( probs - class_to_onehot(Y_))   #NxC
		
		grad_W2 = np.dot(h1.T,dL_ds2)                   #hxN * NxC = hxC
		grad_b2 = np.sum(dL_ds2,axis=0,keepdims=True)   #1xC

		# backward pass
		dL_dh1 = np.dot(dL_ds2,W2.T)                    #NxC * Cxh = Nxh
		
		# backprop through ReLU
		dL_ds1 = dL_dh1
		dL_ds1[h1 <= 0] = 0
		
		grad_W1 = np.dot(X.T,dL_ds1)                    #DxN * Nxh = Dxh
		grad_b1 = np.sum(dL_ds1,axis=0,keepdims=True)   #1xh
		
		# regularization
		grad_W2 += reg*W2
		grad_W1 += reg*W1
		
		# updating parameters
		W1 += -param_delta * grad_W1
		b1 += -param_delta * grad_b1
		W2 += -param_delta * grad_W2
		b2 += -param_delta * grad_b2
		
	return W1,b1,W2,b2
  
def fcann2_classify(X,W1,b1,W2,b2):
	"""Arguments:
		- X: actual datapoints[NxD]
		- W1, b1, W2, b2 - learned parameters
		Returns:
		- predicted class posibilities [NxC]
	"""
	h1 = ReLU(np.dot(X,W1)+b1)	#hidden layer input Nxh
	s2 = np.dot(h1,W2)+b2                        #NxC
	#class probabilities
	exp_s2 = np.exp(s2-np.max(s2))               #NxC
	sumexp = np.vstack(np.sum(exp_s2,axis = 1))
	#log values of class probabilities
	probs = exp_s2/sumexp                        #NXC
	return probs	
	
def fcnn2_decfun(X,W1,b1,W2,b2):
	"""
	Decorator for  method fcann2_classify
	"""
	def classify(X):
		return fcann2_classify(X,W1,b1,W2,b2)[:,1]# or .argmax(axis=1)
	return classify
	
if __name__=="__main__":
  np.random.seed(100)
  X,Y_ = data.sample_gmm_2d(6,2,10)
  
  # train model
  W1,b1,W2,b2 = fcann_train(X,Y_,5,10000,0.05,1e-3)
  
  # evaluate model on the training dataset
  probs = fcann2_classify(X,W1,b1,W2,b2)
  
  # predicted classes
  Y = np.argmax(probs,axis=1)
  
  # model evaluation metrics
  accuracy, pr, M= data.eval_perf_multi(Y, Y_)
  print("Accuracy: ",accuracy)
  print("Precision / Recall: ",pr)
  print("Confussion Matrix: ",M)
  
  # graph the decision surface
  decfun = fcnn2_decfun(X,W1,b1,W2,b2)
  bbox=(np.min(X, axis=0), np.max(X, axis=0))
  data.graph_surface(decfun, bbox, offset=0.5)

  # graph the data points
  data.graph_data(X, Y_, Y, special=[])

  # show the plot
  plt.show()



