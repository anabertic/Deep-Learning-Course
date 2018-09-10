import os
import pickle
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import skimage as ski
import skimage.io
import math

DATA_DIR = './data/'
SAVE_DIR = "./tf_out/"
config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}
config['weight_decay'] = 1e-4

valid_loss = []
valid_accuracy = []
train_loss = []
train_accuracy = []

def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict

def get_data(data_dir):
  classes_pickle = os.path.join(DATA_DIR, 'batches.meta')
  class_names = np.array(unpickle(classes_pickle)['label_names'])
  img_height = 32
  img_width = 32
  num_channels = 3
  train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
  train_y = []
  for i in range(1, 6):
    subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
    train_x = np.vstack((train_x, subset['data']))
    train_y += subset['labels']
  train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1)
  train_y = np.array(train_y, dtype=np.int32)

  subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
  test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1).astype(np.float32)
  test_y = np.array(subset['labels'], dtype=np.int32)
  
  valid_size = 5000
  train_x, train_y = shuffle_data(train_x, train_y)
  valid_x = train_x[:valid_size, ...]
  valid_y = train_y[:valid_size, ...]
  train_x = train_x[valid_size:, ...]
  train_y = train_y[valid_size:, ...]
  data_mean = train_x.mean((0,1,2))
  data_std = train_x.std((0,1,2))

  train_x = (train_x - data_mean) / data_std
  valid_x = (valid_x - data_mean) / data_std
  test_x = (test_x - data_mean) / data_std
  return train_x,valid_x,test_x,train_y,valid_y,test_y,data_mean,data_std,class_names

	
def variance_scaling_initializer(shape, fan_in, factor=2.0, seed=None):
  sigma = np.sqrt(factor / fan_in)
  return tf.Variable(tf.truncated_normal(shape, stddev=sigma))
  

def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def max_pool(x, k ,stride):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='SAME')
def fc(x, W, b):
	x =	tf.reshape(x, [-1, W.get_shape().as_list()[0]])
	return tf.nn.relu(tf.add(tf.matmul(x,W),b))
	
def conv_net(x, weights, biases):
    # CIFAR data input is a  vector of 7168 features (32*32*7 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1,32, 32, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['w_cv1'], biases['b_cv1'])
    # Max Pooling (down-sampling)
    conv1 = max_pool(conv1, k = 3, stride = 2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['w_cv2'], biases['b_cv2'])
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k = 3, stride = 2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = fc(conv2,weights['w_fc1'], biases['b_fc1'])
    fc2 = fc(fc1,weights['w_fc2'], biases['b_fc2'])
    fc3 = tf.reshape(fc2, [-1, weights['out'].get_shape().as_list()[0]])
    fc3 = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
    return fc3
 
def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color, linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color, linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color, linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color, linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot_2.pdf')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)
    
def draw_conv_filters(session, layer, epoch, step, name, save_dir):
    weights = session.run(layer).copy()
    num_filters = weights.shape[3]
    num_channels = weights.shape[2]
    k = weights.shape[0]
    assert weights.shape[0] == weights.shape[1]
    weights -= weights.min()
    weights /= weights.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k,c:c+k,:] = weights[:,:,:,i]
    filename = '%s_epoch_%02d_step_%06d.png' % (name, epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img)
    
def train(session,train_x, train_y, valid_x, valid_y, config,weights):
  lr_policy = config['lr_policy']
  batch_size = config['batch_size']
  num_epochs = config['max_epochs']
  save_dir = config['save_dir']
  num_examples = train_x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  
  session.run(tf.initialize_all_variables())
  plot_data = {}
  plot_data['train_loss'] = []
  plot_data['valid_loss'] = []
  plot_data['train_acc'] = []
  plot_data['valid_acc'] = []
  plot_data['lr'] = []
  for epoch_num in range(1, num_epochs + 1):
    if epoch_num in lr_policy:
      solver_config = lr_policy[epoch_num]
    cnt_correct = 0
    #for i in range(num_batches):
    # shuffle the data at the beggining of each epoch
    permutation_idx = np.random.permutation(num_examples)
    train_x = train_x[permutation_idx]
    train_y = train_y[permutation_idx]
    for step in range(num_batches):
      offset = step * batch_size 

      batch_x = train_x[step*batch_size:(step+1)*batch_size, ...]
      batch_y = train_y[step*batch_size:(step+1)*batch_size, ...]
      start_time = time.time()
      logits_val,loss_val,_ = session.run([logits, reg_loss, train_step] ,feed_dict={X: batch_x, Y_: batch_y, lr:solver_config['lr']})
      yp = np.argmax(logits_val, 1)
      yt = batch_y
      cnt_correct += (yp == yt).sum()
      if step % 5 == 0:
        print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch_num, step*batch_size, num_examples, loss_val))
      if step % 100 == 0:
        draw_conv_filters(session, weights['w_cv1'], epoch_num, step, "conv1", SAVE_DIR)
      if step > 0 and step % 50 == 0:
        print("Train accuracy = %.2f" % (cnt_correct / ((step+1)*batch_size) * 100))
    print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))   
    train_acc = cnt_correct / num_examples * 100
    train_loss = loss_val
    valid_acc,valid_loss = evaluate(session,"Validation", valid_x, valid_y, config)

    duration = time.time() - start_time

    plot_data['train_loss'] += [train_loss]
    plot_data['valid_loss'] += [valid_loss]
    plot_data['train_acc'] += [train_acc]
    plot_data['valid_acc'] += [valid_acc]
    plot_data['lr'] += [solver_config]
    plot_training_progress(SAVE_DIR, plot_data)


def evaluate(session, name, x, y, config):
    print("\nRunning evaluation: ", name)
    batch_size = config['batch_size']
    num_examples = x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    cnt_correct = 0
    loss_avg = 0
  
    for i in range(num_batches):
        batch_x = x[i*batch_size:(i+1)*batch_size, ...]
        batch_y = y[i*batch_size:(i+1)*batch_size, ...]
        logits_val,loss_val = session.run([logits,reg_loss],feed_dict={X: batch_x, Y_: batch_y})
        yp = np.argmax(logits_val, 1)
        yt = batch_y
        cnt_correct += (yp == yt).sum()
        loss_avg += loss_val
        #print("step %d / %d, loss = %.2f" % (i*batch_size, num_examples, loss_val / batch_size))
    valid_acc = cnt_correct / num_examples * 100
    loss_avg /= num_batches
    print(name + " accuracy = %.2f" % valid_acc)
    print(name + " avg loss = %.2f\n" % loss_avg)
    confusion = confusion_matrix(yt,yp)
    print('Confussion matrix')
    print(confusion)
    print('Recall')
    print(recall_score(yt,yp,average=None))
    print('Accuracy')
    print(accuracy_score(yt,yp))
    return valid_acc,loss_avg

def draw_image(img, mean, std):
    img *= std
    img += mean
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    ski.io.show()

def worst_images(session,test_x,test_y,mean,std,class_names):
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
        draw_image(w[3], mean, std)
        
if __name__=="__main__":
    
    train_x,valid_x,test_x,train_y,valid_y,test_y,mean,std,class_names = get_data(DATA_DIR)
	
    num_input = 3072 # MNIST data input (img shape: 32x32)
    num_classes = 10

    X = tf.placeholder(tf.float32, [None, 32,32,3])
    Y_ = tf.placeholder(tf.int32, [None,])

   #Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 16 outputs
        'w_cv1': variance_scaling_initializer([5, 5, 3, 16],25*3),
        # 5x5 conv, 16 inputs, 32 outputs
        'w_cv2': variance_scaling_initializer([5, 5, 16, 32],25*16),
        # fully connected, 7*7*32 inputs, 512 outputs
        'w_fc1': variance_scaling_initializer([8*8*32, 256],68*32),
        'w_fc2': variance_scaling_initializer([256,128],256),
        # 512 inputs, 10 outputs (class prediction)
        'out': variance_scaling_initializer([128, num_classes],128)
   }
    biases = {
        'b_cv1': tf.Variable(tf.zeros([16])),
        'b_cv2': tf.Variable(tf.zeros([32])),
        'b_fc1': tf.Variable(tf.zeros([256])),
        'b_fc2': tf.Variable(tf.zeros([128])),
        'out': tf.Variable(tf.zeros([num_classes]))
    }	
    
    logits = conv_net(X,weights,biases)  

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels =Y_))
    reg = 0
    for w in [weights['w_cv1'],weights['w_cv2'],weights['w_fc1']]:
        reg += tf.nn.l2_loss(w)
    reg_loss = loss + config['weight_decay']*reg
    lr = tf.placeholder(tf.float32)
    train_step =  tf.train.GradientDescentOptimizer(lr).minimize(reg_loss)	
    session = tf.Session()
    train(session,train_x,train_y,valid_x,valid_y,config,weights)
    evaluate(session,"Test",test_x,test_y,config)
    worst_images(session,test_x,test_y,mean,std,class_names)
