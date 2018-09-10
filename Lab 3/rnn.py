import numpy as np

def check_none(U,W,b):
	U = self.U if U is None else U
	W = self.W if W is None else W
	b = self.b if b is None else b  
	return U,W,b 
	
def check_none_output(V,c):
	V = self.V if V is None else V
	W = self.c if c is None else c
	return V,c 
	   
	def softmax(s):
		s = np.max(z, axis=2)
		s = s[:, :, np.newaxis]
		e_x = np.exp(z - s)
		div = np.sum(e_x, axis=2)
		div = div[:, :, np.newaxis]  
		return e_x / div
	
class RNN:	
	def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
		self.hidden_size = hidden_size
		self.sequence_length = sequence_length
		self.vocab_size = vocab_size
		self.learning_rate = learning_rate
		
		self.U = np.random.normal(size=[vocab_size, hidden_size], scale=0.01) # ... input projection
		self.W = np.random.normal(size=[hidden_size, hidden_size], scale=0.01) # ... hidden-to-hidden projection
		self.b = np.zeros([1, hidden_size]) # ... input bias

		self.V = np.random.normal(size=[hidden_size, vocab_size], scale=0.01) # ... output projection
		self.c = np.zeros([1, vocab_size]) # ... output bias
        # memory of past gradients - rolling sum of squares for Adagrad
		self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
		self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)
			
	def rnn_step_forward(self, x, h_prev, U, W, b):
		# A single time step forward of a recurrent neural network with a 
		# hyperbolic tangent nonlinearity.

		# x - input data (minibatch size x input dimension)
		# h_prev - previous hidden state (minibatch size x hidden size)
		# U - input projection matrix (input dimension x hidden size)
		# W - hidden to hidden projection matrix (hidden size x hidden size)
		# b - bias of shape (hidden size x 1)
		U,W,b = check_none(U,W,b)
		#(minibatch size x input dimension) x (input dimension x hidden size) + (minibatch size x hidden size)x(hidden size x hidden size)
		h_current = np.tanh(np.dot(x, U) + np.dot(h_prev, W) + b)
		cache = (W, x, h_prev, h_current)
		# return the new hidden state and a tuple of values needed for the backward step
		return h_current, cache


	def rnn_forward(self, x, h0, U, W, b):
		# Full unroll forward of the recurrent neural network with a 
		# hyperbolic tangent nonlinearity

		# x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
		# h0 - initial hidden state (minibatch size x hidden size)
		# U - input projection matrix (input dimension x hidden size)
		# W - hidden to hidden projection matrix (hidden size x hidden size)
		# b - bias of shape (hidden size x 1)
		U,w,b = check_none(U,W,b)
		
		h, cache = [h0], []

		for sequence in x.transpose(1, 0, 2): #30  x  (32x72) po redu svako slovo svakog sequenca svakog batcha
			h_current, cache_current = self.rnn_step_forward(sequence, h[-1], U, W, b)
			h.append(h_current)
			cache.append(cache_current)
    
		# Skip initial hidden state
		# return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step
		return np.array(h[1:]).transpose(1, 0, 2),cache

	def rnn_step_backward(self, grad_next, cache):
		# A single time step backward of a recurrent neural network with a 
		# hyperbolic tangent nonlinearity.

		# grad_next - upstream gradient of the loss with respect to the next hidden state and current output
		# cache - cached information from the forward pass

		# compute and return gradients with respect to each parameter
		# HINT: you can use the chain rule to compute the derivative of the
		# hyperbolic tangent function and use it to compute the gradient
		# with respect to the remaining parameters
		W, x, h_prev, h_current= cache

		# compute and return gradients with respect to each parameter
		da = (1 - h_current**2) * grad_next #derivative tanh * dL/dh
		dh_prev = np.dot(da, np.transpose(self.W))
		dU = (1.0/grad_next.shape[0])*np.dot(x.T, da)
		dW = (1.0/grad_next.shape[0])*np.dot(h_prev.T, da)
		db = (1.0/grad_next.shape[0])*np.sum(da, axis=0) 

		return dh_prev, dU, dW, db

	def rnn_backward(self, dh, cache):
		# Full unroll forward of the recurrent neural network with a 
		# hyperbolic tangent nonlinearity
        
		dU, dW, db = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.b)

		# compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?
		dh_prev = np.zeros_like(dh[-1])
		for dh_current, cache_current in reversed(list(zip(dh, cache))):
			dh_prev, dU_curr, dW_curr, db_curr = self.rnn_step_backward(dh_prev + dh_current, cache_current)
			dU += dU_curr
			dW += dW_curr
			db += db_curr
		return np.clip(dU, -5, 5), np.clip(dW, -5, 5), np.clip(db, -5, 5)
	
	def output(self,h, V, c):
		# Calculate the output probabilities of the network
		return softmax(np.dot(h, V) + c)

	def output_loss_and_grads(self, h, V, c, y):
		# Calculate the loss of the network for each of the outputs
    
		# h - hidden states of the network for each timestep. 
		#     the dimensionality of h is (batch size x sequence length x hidden size (the initial state is irrelevant for the output)
		# V - the output projection matrix of dimension hidden size x vocabulary size
		# c - the output bias of dimension vocabulary size x 1
		# y - the true class distribution - a tensor of dimension 
		#     batch_size x sequence_length x vocabulary size - you need to do this conversion prior to
		#     passing the argument. A fast way to create a one-hot vector from
		#     an id could be something like the following code:

		#   y[batch_id][timestep] = np.zeros((vocabulary_size, 1))
		#   y[batch_id][timestep][batch_y[timestep]] = 1

		#     where y might be a list or a dictionary.

		V,c = check_none_output(V,c)
		
		loss, dh, dV, dc = 0.0, [], np.zeros_like(self.V), np.zeros_like(self.c)
		batch_size = h.shape[0]

		for t in range(self.sequence_length):
			yp,h_t = y[:, t, :],h[:, t, :]
			# calculate the output (o) - unnormalized log probabilities of classes
			# calculate yhat - softmax of the output           
			yhat = self.output(h_t, V, c)
			# calculate the cross-entropy loss
			loss += -(1.0/batch_size)*np.sum(np.log(yhat)*yp)
			# calculate the derivative of the cross-entropy softmax loss with respect to the output (o)
			dO = (1.0/batch_size)*(yhat - yp)
			# calculate the gradients with respect to the output parameters V and c
			dV += np.dot(np.transpose(h_t), dO)
			dc += np.sum(dO, axis=0)
			# calculate the gradients with respect to the hidden layer h
			dh.append(np.dot(dO, np.transpose(V)))
			return loss, dh, np.clip(dV,-5,5), np.clip(dc,-5,5)
			# The inputs to the function are just indicative since the variables are mostly present as class properties

	def update(self, dU, dW, db, dV, dc,eps):
		# update memory matrices
		self.memory_U += np.square(dU)
		self.memory_W += np.square(dW)
		self.memory_b += np.square(db)
		self.memory_V += np.square(dV)
		self.memory_c += np.square(dc)
        
        # perform the Adagrad update of parameters
		self.U -= self.learning_rate * dU / np.sqrt(self.memory_U + eps)
		self.W -= self.learning_rate * dW / np.sqrt(self.memory_W + eps)
		self.b -= self.learning_rate * db / np.sqrt(self.memory_b + eps)
		self.V -= self.learning_rate * dV / np.sqrt(self.memory_V + eps)
		self.c -= self.learning_rate * dc / np.sqrt(self.memory_c + eps)


	def step(self, h0, x_oh, y_oh):
		h, cache = self.rnn_forward(x_oh, h0,self.U,self.W,self.b)
		loss, dh, dV, dc = self.output_loss_and_grads(h,self.V,self.c, y_oh)
		dU, dW, db = self.rnn_backward(dh, cache)

		self.update(dU, dW, db, dV, dc,1e-6)
		return loss, h[:, -1, :]

