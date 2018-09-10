import tensorflow as tf
import numpy as np
import data
from data import *
import logreg
from logreg import *
import sklearn
from sklearn.model_selection import train_test_split

class TFDeep:
	
	def __init__(self, neurons, param_delta = 0.1, param_lambda=0.0001, activation=tf.nn.relu, optimizer=tf.train.GradientDescentOptimizer,decay=False):
		"""Arguments:
			- neurons - model configuration [D , ..(# of neurons in each layer).. , C]
			- param_delta: training step
			- param lambda: regularization coefficient
			- activation: activation function used in model
			- optimizer: optimizer used during training
			- decay: option in case AdamOptimizer is used
		"""
		# data dimensionality
		D = neurons[0]  # dimensions of each sample
		C = neurons[-1] # number of classes
		
		# data and parameters
		self.X = tf.placeholder(tf.float32, [None,D])
		self.Yoh_ = tf.placeholder(tf.float32, [None,C])
		
		self.W = []
		self.b = []
		self.h = self.X
		self.reg = []
		
		# for decay
		global_step = None
		
		for i in range(len(neurons[:-1])):
			self.W.append(tf.Variable(tf.random_normal(shape=[neurons[i], neurons[i+1]]),name='W_%s' %(i+1)))
			self.b.append(tf.Variable(tf.zeros((1, neurons[i+1])), tf.float32,name ='b_%s' %(i+1)))
			self.reg.append(tf.nn.l2_loss(self.W[i]))

		for i in range(len(self.W)-1):
			self.h = activation(tf.matmul(self.h,self.W[i])+self.b[i])

		# model
		self.probs = tf.nn.softmax(tf.matmul(self.h,self.W[-1])+self.b[-1])
		
		# loss
		self.loss = tf.reduce_mean(-tf.reduce_sum(self.Yoh_*tf.log(tf.clip_by_value(self.probs, 1e-10, 1.0)),reduction_indices=1))
		self.reg_loss = self.loss + param_lambda*np.sum(self.reg)
		
		if decay:
			global_step = tf.Variable(0, trainable=False)
			param_delta = tf.train.exponential_decay(param_delta, global_step, decay_steps=1, decay_rate=1-1e-4, staircase=True)
        		
        # training 
		self.train_step = optimizer(param_delta).minimize(self.reg_loss,global_step=global_step)

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
			loss,_,W,b,h= self.Session.run([self.reg_loss,self.train_step,self.W,self.b,self.h], feed_dict = {self.X:X, self.Yoh_:Yoh_})
			if (i%100==0):
				print(i," loss: ",loss)
			
	def early_stop(self,X,Yoh_,X_val,Yoh_val,param_niter):
		""" Method for early stopping """
		min_loss = 100000
		self.Session.run(tf.initialize_all_variables())
		W = self.W
		b = self.b
		for i in range(param_niter):
			loss,_ = self.Session.run([self.reg_loss,self.train_step],feed_dict={self.X : X, self.Yoh_:Yoh_})
			if i % 100 == 0:
				val_loss = self.Session.run([self.reg_loss],feed_dict={self.X:X_val,self.Yoh_:Yoh_val})[0]
				print(i," loss:",loss,", validation loss:",val_loss)
				if val_loss < min_loss:
					print("Best loss found")
					min_loss = val_loss
					W = self.W
					b = self.b
		self.W,self.b = W,b
	
	def train_mb(self, X, Yoh_, param_niter, n, epochs):
		""" Method for stohastic gradient using minibatches """
		self.Session.run(tf.initialize_all_variables())
		for e in range(epochs):
			print("Epoch: ",e+1)
			batches_X,batches_Y = shuffle_and_slice(X,Yoh_,n)
			for k in range(param_niter):
				for i in range(n):
					loss,_ = self.Session.run([self.reg_loss,self.train_step],feed_dict={self.X:batches_X[i], self.Yoh_:batches_Y[i]})	
					if (k%100==0):
						print(k," Batch ",i,", loss: ",loss)
	
	def eval(self,X):
		"""Arguments:
		- X: actual datapoints[NxD]
		Returns:
		- predicted class posibilities [NxC]
		"""
		probs = self.Session.run([self.probs], feed_dict={self.X:X})
		return probs
	
	def count_params(self):
		""" Method for counting all trainable parameters """
		variables_names = [v.name for v in tf.trainable_variables()]
		values = self.Session.run(variables_names)
		sum = 0
		for name,v in zip(variables_names,values):
			print("Variable: ", name,v)
			sum+=v.shape[0]*v.shape[1]
		print(sum)
	
	def get_W(self):
		return self.Session.run(self.W)

	
def shuffle_and_slice(X,Yoh_,n):
	""" Method for creating mini batches from data
		Arguments: 
			X - data [NxD]
			Yoh_ - one hot vector representing data labels [NxC] 
			n - number of batches 
		Returns:
			batches_X - numpy array of batches made from X
			batches_Y - numpy array of corresponding labels made from Yoh_
	"""
	indices = [i for i in range(X.shape[0])]
	random.shuffle(indices)
	ind = np.asarray(indices).reshape(n,-1)
	batches_X = []
	batches_Y = []
	for i,row in enumerate(ind):
		batch_X = []
		batch_Y = []
		for j in list(row):
			batch_X.append(X[j])
			batch_Y.append(Yoh_[j])
		batches_X.append(np.asarray(batch_X).reshape(-1,X.shape[1]))
		batches_Y.append(np.asarray(batch_Y).reshape(-1,Yoh_.shape[1]))
	return batches_X,batches_Y		

if __name__=="__main__":
	
	# random number generator
	np.random.seed(100)
	tf.set_random_seed(100)
	
	# instantiating data
	X,Y_ = sample_gmm_2d(6,2,10)
	#or X,Y_ = data.sample_gauss(3,100)

	Yoh_ = class_to_onehot(Y_)
	
	# building graph
	tlfr = TFDeep([2,10,10,2])   # 2 features, 10 neurons in each layer, 2 output classes

	# normal training
	tlfr.train(X,Yoh_,10000)

	#early stopping
	X_train, X_valid, y_train, y_valid = train_test_split(X, Yoh_, test_size=0.2, random_state=42)
	#tlfr.early_stop(X_train,y_train,X_valid,y_valid,10000)
	
	# stohastic minibatch
	#tlfr.train_mb(X,Yoh_,1000,5,500)
	
	# fetching probabilities on train set
	probs = tlfr.eval(X)
	Y = np.argmax(probs[0],axis=1)
	
	accuracy, pr, M= data.eval_perf_multi(Y, Y_)
	print("Accuracy: ",accuracy)
	print("Precision / Recall: ",pr)
	print("Confussion Matrix: ",M)
	tlfr.count_params()

	#graph the decision surface
	decfun = lambda x: tlfr.eval(x)[0][:,1]
	bbox=(np.min(X, axis=0), np.max(X, axis=0))
	data.graph_surface(decfun, bbox, offset=0.5)

	# graph the data points
	data.graph_data(X, Y_, Y, special=[])
	
	# show the plot
	plt.show()
