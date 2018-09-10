import tensorflow as tf
import matplotlib.pyplot as plt
import tf_deep
from tensorflow.examples.tutorials.mnist import input_data
from ksvm_wrap import SVMWrapper

tf.app.flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
mnist = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=True)

N = mnist.train.images.shape[0]
D=mnist.train.images.shape[1]
C=mnist.train.labels.shape[1]
"""
images = []
for i in mnist.train.images:
	images.append(i.reshape(28,28))
print(images[0].shape)
plt.imshow(images[0],cmap=plt.get_cmap('gray'),vmin=0,vmax=1)
plt.show()
"""

# [784, 10]
tflr = tf_deep.TFDeep([784,10])
tflr.train(mnist.train.images,mnist.train.labels,1000)

W = tflr.get_W()
plt.figure(figsize=(12,8))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(W[0][:,i].reshape(28,28), cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.show()

#performanse na skupu za ucenje
train_probs = tflr.eval(mnist.train.images)
train_Y = np.argmax(train_probs[0],axis = 1)
train_Y_ = np.argmax(mnist.train.labels,axis=1)
accuracy, pr, M = data.eval_perf_multi(train_Y, train_Y_)
print("Accuracy: ",accuracy)
print("Precision / Recall: ",pr)
print("Confussion Matrix: \n ",M)


#performanse na skupu za testiranje
test_probs = tflr.eval(mnist.test.images)
test_Y = np.argmax(test_probs[0],axis = 1)
test_Y_ = np.argmax(mnist.test.labels,axis=1)
accuracy, pr, M = data.eval_perf_multi(test_Y, test_Y_)
print("Accuracy: ",accuracy)
print("Precision / Recall: ",pr)
print("Confussion Matrix: \n ",M)

# [784,100,10]
tflr_2 = tf_deep.TFDeep([784,100,10],param_delta=0.5)
tflr_2.train(mnist.train.images,mnist.train.labels,param_niter=5000)

train_probs = tflr_2.eval(mnist.train.images)
#train_Y = np.argmax(train_probs[0],axis = 1)
print(train_probs[0].shape)
train_Y = np.argmax(train_probs[0],axis = 1)
train_Y_ = np.argmax(mnist.train.labels,axis=1)
accuracy, pr, M = data.eval_perf_multi(train_Y, train_Y_)
print("Accuracy: ",accuracy)
print("Precision / Recall: ",pr)
print("Confussion Matrix: \n ",M)

test_Y = np.argmax(test_probs[0],axis = 1)
test_Y_ = np.argmax(mnist.test.labels,axis=1)
accuracy, pr, M = data.eval_perf_multi(test_Y, test_Y_)
print("Accuracy: ",accuracy)
print("Precision / Recall: ",pr)
print("Confussion Matrix: \n ",M)

# Early stopping
X_train, X_valid, y_train, y_valid = train_test_split(mnist.train.images, mnist.train.labels, test_size=0.2, random_state=42)
tflr_val = tf_deep.TFDeep([784,10])
tflr_val.early_stop(X_train,y_train,X_valid,y_valid,5000)

# Mini Batch
tflr_mb = tf_deep.TFDeep([784,10])
tflr_mb.train_mb(mnist.train.images,mnist.train.labels,n = 200,epochs = 500,param_niter=5000)

# Adam Optimizer
tflr_adam = tf_deep.TFDeep([784,10],optimizer=tf.train.AdamOptimizer,param_delta=0.0001)
tflr_adam.train(mnist.train.images,mnist.train.labels,param_niter=5000)

# Adam Optimizer with decay
tflr_adam_decay = tf_deep.TFDeep([784,10],optimizer=tf.train.AdamOptimizer,param_delta=0.0001,decay=True)
tflr_adam_decay.train(mnist.train.images,mnist.train.labels,param_niter=5000)

# SVM
np.random.seed(100)
model = SVMWrapper(mnist.train.images, mnist.train.labels.argmax(axis=1)c=1, g='auto')


