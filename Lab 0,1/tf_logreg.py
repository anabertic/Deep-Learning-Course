import tensorflow as tf
import numpy as np
import data
from data import *
import logreg
from logreg import *

class TFLogreg:
	def __init__(self, D, C, param_delta = 0.1, param_lambda=0.25):
		"""Arguments:
			- D: dimensions of each datapoint
			- C: number of classes
			- param_delta: training step
		"""
		
		# data and parameter definition:
		self.X = tf.placeholder(tf.float32, [None,D])
		self.Yoh_ = tf.placeholder(tf.float32, [None,C])
		self.W =  tf.Variable(tf.random_normal(shape=[D, C]))
		self.b = tf.Variable(tf.zeros((1, C)), tf.float32)
		
		# model 
		self.probs = tf.nn.softmax(tf.matmul(self.X,self.W) + self.b)
		
		# loss
		self.loss = tf.reduce_mean(-tf.reduce_sum(self.Yoh_*tf.log(self.probs),reduction_indices=1))
		self.reg_loss = self.loss + param_lambda*tf.nn.l2_loss(self.W)
		
		# learning step
		self.train_step = tf.train.GradientDescentOptimizer(param_delta).minimize(self.reg_loss)
		
		self.Session = tf.Session()

		
	def train(self, X, Yoh_, param_niter):
		"""Arguments:
			- X: actual datapoints [NxD]
			- Yoh: one-hot encoded labels [NxC]
			- param_niter: number of iterations
		"""
		# parameter initialization
		self.Session.run(tf.initialize_all_variables())
	 
		# optimization loop
		for i in range(param_niter):
			loss,_,W,b = self.Session.run([self.reg_loss,self.train_step,self.W,self.b], feed_dict = {self.X:X, self.Yoh_:Yoh_})
			if (i%100==0):
				print(i," loss: ",loss)

	def eval(self,X):
		"""Arguments:
		- X: actual datapoints[NxD]
		Returns:
		- predicted class posibilities [NxC]
		"""
		probs = self.Session.run([self.probs], feed_dict={self.X:X})
		return probs
		
if __name__=="__main__":
	
	# random data generator
	np.random.seed(100)
	tf.set_random_seed(100)
	
	X,Y_ = data.sample_gauss(3,100)
	Yoh_ = class_to_onehot(Y_)
	
	#building graph
	tlfr = TFLogreg(X.shape[1],Yoh_.shape[1],0.1)

	# learning parameters
	tlfr.train(X,Yoh_,1000)

	# fetch probabilities on train set
	probs = tlfr.eval(X)
	Y = np.argmax(probs[0],axis=1)
	
	accuracy, pr, M= data.eval_perf_multi(Y, Y_)
	print("Accuracy: ",accuracy)
	print("Precision / Recall: ",pr)
	print("Confussion Matrix: ",M)
  
	#graph the decision surface
	decfun = logreg.logreg_decfun(X,W,b)
	bbox=(np.min(X, axis=0), np.max(X, axis=0))
	data.graph_surface(decfun, bbox, offset=0.5)

	# graph the data points
	data.graph_data(X, Y_, Y, special=[])

	# show the plot
	plt.show()
