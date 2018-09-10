import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

## 1. computing graph definition
# data and parameters
X = tf.placeholder(tf.float32, [None,1])
Y_ = tf.placeholder(tf.float32, [None,1])
a = tf.Variable(0.0)
b = tf.Variable(0.0)
Xt = np.asarray([1,2,3]).reshape(-1,1)
Y_t = np.asarray([1,2,3]).reshape(-1,1)
N = Y_t.shape[0]
# afine regression model
Y = a * X + b

# quadratic loss
loss = (1.0/(2*N))*(Y-Y_)**2

# ptimization: gradient descent
trainer = tf.train.GradientDescentOptimizer(0.1)
#train_op = trainer.minimize(loss) # node for one descent iteration 

grads_vars = trainer.compute_gradients(loss, [a, b])
optimize = trainer.apply_gradients(grads_vars)

# another printout
#grads_vars = tf.Print(grads_vars, [grads_vars], 'Status:')

# analitic gradient computing (for comparing purposes)
grad_a = (1.0/N)*tf.matmul((Y-Y_),X,transpose_a = True)
grad_b = (1.0/N)*(Y-Y_)

## 2. parameter initialization
sess = tf.Session()
sess.run(tf.initialize_all_variables())

## 3. training
for i in range(1000):
	val_loss, val_grads, val_grad_a,val_grad_b = sess.run([loss,grads_vars,grad_a,grad_b],feed_dict = {X:Xt, Y_:Y_t})
	val_loss,_,val_a,val_b = sess.run([loss,optimize,a,b], feed_dict = {X:Xt, Y_:Y_t})
	if (i%100==0):
		print(i," loss: ",val_loss.sum())
		print("Computed gradients: (",val_grad_a[0][0],", ",val_a,") , (",val_grad_b.sum(),", ",val_b,")")
		print("Tensorflow gradients ",val_grads)
