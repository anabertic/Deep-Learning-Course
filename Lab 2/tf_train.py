import time
import math
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import skimage as ski
import skimage.io
import os
DATA_DIR = './mnist/'
SAVE_DIR = "./tf_out/"
config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}
config['weight_decay'] = 1e-4

def get_data(data_dir):
    np.random.seed(int(time.time() * 1e6) % 2**31)
    dataset = input_data.read_data_sets(data_dir, one_hot=True)
    train_x = dataset.train.images
    train_x = train_x.reshape([-1, 28, 28,1])
    train_y = dataset.train.labels
    valid_x = dataset.validation.images
    valid_x = valid_x.reshape([-1, 28, 28,1])
    valid_y = dataset.validation.labels
    test_x = dataset.test.images
    test_x = test_x.reshape([-1, 28, 28, 1])
    test_y = dataset.test.labels
    train_mean = train_x.mean()
    train_x -= train_mean
    valid_x -= train_mean
    test_x -= train_mean
    return train_x,valid_x,test_x,train_y,valid_y,test_y
	
def variance_scaling_initializer(shape, fan_in, factor=2.0, seed=None):
  sigma = np.sqrt(factor / fan_in)
  return tf.Variable(tf.truncated_normal(shape, stddev=sigma))
  

def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def max_pool(x, k = 2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
	
def conv_net(x, weights, biases):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['w_cv1'], biases['b_cv1'])
    # Max Pooling (down-sampling)
    conv1 = max_pool(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['w_cv2'], biases['b_cv2'])
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['w_fc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['w_fc1']), biases['b_fc1'])
    fc1 = tf.nn.relu(fc1)
    fc2 = tf.reshape(fc1, [-1, weights['out'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return fc2
 

def draw_conv_filters(session, epoch, step, weights, save_dir, name):
  weights_val = session.run(weights)
  k = weights_val.shape[0]
  C = weights_val.shape[2]
  num_filters = weights_val.shape[3]
  w = weights_val.copy()
  w = w.reshape(num_filters, C, k, k)
  w -= w.min()
  w /= w.max()
  
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  #for i in range(C):
  
  for i in range(1):
    img = np.zeros([height, width])
    for j in range(num_filters):
      r = int(j / cols) * (k + border)
      c = int(j % cols) * (k + border)
      img[r:r+k,c:c+k] = w[j,i]
    filename = '%s_epoch_%02d_step_%06d_input_%03d.png' % (name, epoch, step, i)
    ski.io.imsave(os.path.join(save_dir, filename), img)

def worst_images(session,test_x,test_y):
    worst = []
    outputs = tf.nn.softmax(logits).eval(feed_dict={X: test_x, Y_: test_y}, session=session)
    losses = reg_loss.eval(feed_dict={X: test_x, Y_: test_y}, session=session)
    for o in range(len(outputs)):
        if (np.argmax(outputs[o])!=test_y[o]):
            worst.append((outputs[o],np.argmax(outputs[o]),test_y[o],test_x[o]))
        worst = worst[-20:]
        for w in worst:
            print('Predicted class ',class_names[w[1]])
            print('Correct class ',class_names[w[2]])
            print('Classes with max prediction')
            print(class_names[w[0].argsort()[-3:][::-1]])

        
def train(session,train_x, train_y, valid_x, valid_y, config,weights):
  lr_policy = config['lr_policy']
  batch_size = config['batch_size']
  max_epochs = config['max_epochs']
  save_dir = config['save_dir']
  num_examples = train_x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  
  session.run(tf.initialize_all_variables())
  
  for epoch in range(1, max_epochs+1):
    if epoch in lr_policy:
      solver_config = lr_policy[epoch]
    cnt_correct = 0
    #for i in range(num_batches):
    # shuffle the data at the beggining of each epoch
    permutation_idx = np.random.permutation(num_examples)
    train_x = train_x[permutation_idx]
    train_y = train_y[permutation_idx]
    #for i in range(100):
    for i in range(num_batches):
      # store mini-batch to ndarray
      batch_x = train_x[i*batch_size:(i+1)*batch_size, :]
      batch_y = train_y[i*batch_size:(i+1)*batch_size, :]
      
      logits_val,loss_val,_ = session.run([logits, reg_loss, train_step] ,feed_dict={X: batch_x, Y_: batch_y, lr:solver_config['lr']})
      
      # compute classification accuracy
      yp = np.argmax(logits_val, 1)
      yt = np.argmax(batch_y, 1)
      cnt_correct += (yp == yt).sum()

      if i % 5 == 0:
        print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, num_examples, loss_val))
      if i % 100 == 0:
        draw_conv_filters(session,epoch, i*batch_size, weights['w_cv1'],save_dir,'cv1')
      if i > 0 and i % 50 == 0:
        print("Train accuracy = %.2f" % (cnt_correct / ((i+1)*batch_size) * 100))
    print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
    evaluate(session,"Validation", valid_x, valid_y, config)


def evaluate(session, name, x, y, config):
  print("\nRunning evaluation: ", name)
  batch_size = config['batch_size']
  num_examples = x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  cnt_correct = 0
  loss_avg = 0
  
  for i in range(num_batches):
    batch_x = x[i*batch_size:(i+1)*batch_size, :]
    batch_y = y[i*batch_size:(i+1)*batch_size, :]
    logits_val,loss_val = session.run([logits,reg_loss],feed_dict={X: batch_x, Y_: batch_y})
    yp = np.argmax(logits_val, 1)
    yt = np.argmax(batch_y, 1)
    cnt_correct += (yp == yt).sum()
    loss_avg += loss_val
    #print("step %d / %d, loss = %.2f" % (i*batch_size, num_examples, loss_val / batch_size))
  valid_acc = cnt_correct / num_examples * 100
  loss_avg /= num_batches
  print(name + " accuracy = %.2f" % valid_acc)
  print(name + " avg loss = %.2f\n" % loss_avg)

if __name__=="__main__":
    
    train_x,valid_x,test_x,train_y,valid_y,test_y = get_data(DATA_DIR)
	
    num_input = 784 # MNIST data input (img shape: 28x28)
    num_classes = 10

    X = tf.placeholder(tf.float32, [None, 28,28,1])
    Y_ = tf.placeholder(tf.float32, [None, num_classes])

   #Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 16 outputs
        'w_cv1': variance_scaling_initializer([5, 5, 1, 16],25),
        # 5x5 conv, 16 inputs, 32 outputs
        'w_cv2': variance_scaling_initializer([5, 5, 16, 32],25*16),
        # fully connected, 7*7*32 inputs, 512 outputs
        'w_fc1': variance_scaling_initializer([7*7*32, 512],49*32),
        # 512 inputs, 10 outputs (class prediction)
        'out': variance_scaling_initializer([512, num_classes],512)
   }
    biases = {
        'b_cv1': tf.Variable(tf.zeros([16])),
        'b_cv2': tf.Variable(tf.zeros([32])),
        'b_fc1': tf.Variable(tf.zeros([512])),
        'out': tf.Variable(tf.zeros([num_classes]))
    }	
    
    logits = conv_net(X,weights,biases)  

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels =Y_))
    reg = 0
    for w in [weights['w_cv1'],weights['w_cv2'],weights['w_fc1']]:
        reg += tf.nn.l2_loss(w)
    reg_loss = loss + config['weight_decay']*reg
    lr = tf.placeholder(tf.float32)
    train_step =  tf.train.GradientDescentOptimizer(lr).minimize(reg_loss)	
    session = tf.Session()
    train(session,train_x,train_y,valid_x,valid_y,config,weights)
    evaluate(session,"Test",test_x,test_y,config)
