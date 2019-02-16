import numpy as np
import os
from nn import *


def forward_pass(X, W, b, layer, activation=Sigmoid()):
    pre_act = X @ W + b
    post_act = activation.forward(pre_act)
    params[layer] = (X, pre_act, post_act)
    return post_act

def get_random_batches(x,y,batch_size):
    batches = []
    N = x.shape[0]
    rand_index = np.random.permutation(N)
    num_batches = N//batch_size
    for i in range(num_batches):
        index = rand_index[i*batch_size:(i+1)*batch_size]
        x_batch = x[index,:]
        y_batch = y[index,:]
        batches.append((x_batch,y_batch))
    return batches

def BN_forward(x,gamma,beta,layer):
    mean = np.mean(x, axis=0)
    variance_pre = (x-mean)*(x-mean)
    variance = np.mean(variance_pre,axis=0)
    nomin = np.array(list(map(lambda data: 1/(np.sqrt(data+epsilon)), variance)))
    nomin = np.tile(nomin, (x.shape[0], 1))
    x_norma = (x-mean)*nomin
    cache[layer] = {x, x_norma, mean, variance, nomin, gamma, beta}
    return gamma*x_norma+beta

def BN_backward(out, cache, layer):
    x, x_norma, mean, variance, nomin, gamma, beta = cache[layer]
    dX_norm = out * gamma
    dvariance = -np.sum(dX_norm * (x - mean), axis=0) * nomin**3 /2
    dmean = np.sum(dX_norm * -nomin, axis=0) + dvariance * np.mean(-2 * (x - mean), axis=0)
    dX = (dX_norm * nomin) + (dvariance * 2 * (x - mean) / x.shape[0]) + (dmean / x.shape[0])
    dgamma = np.sum(out * x_norma, axis=0)
    dbeta = np.sum(out, axis=0)

    return dX, dgamma, dbeta


def compute_loss_and_acc(y, probs):
    probs = probs.copy()
    log = lambda p: np.log(p)
    log_probs = log(probs)
    cross = y*log_probs
    loss = -np.sum(cross)
    n = y.shape[0]
    index = 0
    y_label = np.argmax(y, axis=1)
    probs_label = np.argmax(probs, axis=1)

    for i in range(n):
        if y_label[i] == probs_label[i]:
            index += 1
    acc = index/n
    return loss, acc


def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def linear_deriv(post_act):
    return np.ones_like(post_act)

def backwards(delta, W, b, layer, activation_deriv=sigmoid_deriv):
    X, pre_act, post_act = params[layer]
    delta_pre = delta * activation_deriv(post_act)
    grad_W = X.transpose() @ delta_pre
    grad_b = np.sum(delta_pre, axis=0, keepdims=True)
    grad_X = delta_pre @ W.transpose()
    return grad_X, grad_W, grad_b

def get_val_loss(val_X, val_y, val_loss, W1, b1, W2, b2, W3, b3, val_size, val_acc):
    # forward
    h1 = forward_pass(val_X, W1, b1, layer=1, activation=Sigmoid())
    h = forward_pass(h1, W2, b2, layer=2, activation=Sigmoid())
    p = forward_pass(h, W3, b3, layer=3, activation=Softmax())

    # calculate loss
    loss, acc = compute_loss_and_acc(val_y, p)
    val_loss.append(loss/val_size)
    val_acc.append(acc)

def test_loss_and_accuracy(test_X, test_y, W1, b1, W2, b2, W3, b3, test_size):
    h1 = forward_pass(test_X, W1, b1, layer=1, activation=Sigmoid())
    h = forward_pass(h1, W2, b2, layer=2, activation=Sigmoid())
    p = forward_pass(h, W3, b3, layer=3, activation=Softmax())

    loss, acc = compute_loss_and_acc(test_y, p)
    return loss/test_size, acc


input_size = 1568
hidden_size = 100
hidden_size1 = 20
output_size = 19
batch_size = 100
max_iters = 150
learning_rate = 1e-2
params = {1:{},2:{},3:{}}
cache = {1:{},2:{},3:{}}
Momentum = -0.0
weight_decay = 0.0
epsilon = 1e-9

'''
    Get the train/val/test dataset
'''
train = np.loadtxt('../data/data/train.txt',delimiter= ',',unpack= False)
train_X = train[:,:-1]
train_Y = train[:,-1].astype(int)
train_y = np.zeros((train_Y.shape[0],19))
train_y[np.arange(train_Y.shape[0]),train_Y] = 1
print("Successfully loaded training data")

val = np.loadtxt('../data/data/val.txt',delimiter= ',',unpack= False)
val_X = val[:,:-1]
val_Y = val[:,-1].astype(int)
val_y = np.zeros((val_Y.shape[0],19))
val_y[np.arange(val_Y.shape[0]),val_Y] = 1
print("Successfully loaded val data")

test = np.loadtxt('../data/data/test.txt',delimiter= ',',unpack= False)
test_X = test[:,:-1]
test_Y = test[:,-1].astype(int)
test_y = np.zeros((test_Y.shape[0],19))
test_y[np.arange(test_Y.shape[0]),test_Y] = 1
print("Successfully loaded test data")

train_size = train_X.shape[0]
val_size = val_X.shape[0]
test_size = test_X.shape[0]
print("train_size:", train_size)
print("val_size:", val_size)
print("test_size:", test_size)



'''
    Initialize the weight
'''

W1 = random_normal_weight_init(input_size, hidden_size)
b1 = zeros_bias_init(hidden_size)
W2 = random_normal_weight_init(hidden_size, hidden_size1)
b2 = zeros_bias_init(hidden_size1)
W3 = random_normal_weight_init(hidden_size1, output_size)
b3 = zeros_bias_init(output_size)
M_W1, M_W2, M_W3, M_b1, M_b2, M_b3 = W1, W2, W3, b1, b2, b3
print("Weight initialized")


'''
    Get batches
'''
batches = get_random_batches(train_X,train_y,batch_size)
print("Successfully splited the training data")


'''
    Train the network
'''

train_loss = []
val_loss = []
train_acc = []
val_acc = []
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0
    for xb, yb in batches:
        # forward
        h1 = forward_pass(xb, W1, b1, layer=1, activation=Sigmoid())
        h2 = forward_pass(h1, W2, b2, layer=2, activation=Sigmoid())
        p = forward_pass(h2, W3, b3, layer=3, activation=Softmax())

        # calculate loss
        loss, acc = compute_loss_and_acc(yb, p)
        # print(loss)

        total_loss += loss
        avg_acc += acc

        # backward
        delta1 = p - yb
        delta2, grad_W3, grad_b3 = backwards(delta1, W3, b3, layer=3, activation_deriv=linear_deriv)
        delta3, grad_W2, grad_b2 = backwards(delta2, W2, b2, layer=2, activation_deriv=sigmoid_deriv)
        _, grad_W1, grad_b1 = backwards(delta3, W1, b1, layer=1, activation_deriv=sigmoid_deriv)


        # Update the weight

        '''
            With Momentum, with weight decay
        '''
        M_W1 = Momentum * M_W1 - (1+Momentum) * learning_rate * grad_W1 - learning_rate * np.abs(weight_decay * M_W1)
        W1 += M_W1
        M_W2 = Momentum * M_W2 - (1+Momentum) * learning_rate * grad_W2 - learning_rate * np.abs(weight_decay * M_W2)
        W2 += M_W2
        M_W3 = Momentum * M_W3 - (1+Momentum) * learning_rate * grad_W3 - learning_rate * np.abs(weight_decay * M_W3)
        W3 += M_W3

        M_b1 = Momentum * M_b1 - (1+Momentum) * learning_rate * grad_b1
        b1 += M_b1
        M_b2 = Momentum * M_b2 - (1+Momentum) * learning_rate * grad_b2
        b2 += M_b2
        M_b3 = Momentum * M_b3 - (1+Momentum) * learning_rate * grad_b3
        b3 += M_b3


    if itr % 2 == 0:
        np.save("./saved_model2/W3_%d.npy" % itr, W3)
        avg_acc /= len(batches)
        train_acc.append(avg_acc)
        print("Training epoch:", itr, "training accuracy:", avg_acc)
        train_loss.append(total_loss/train_size)
        get_val_loss(val_X, val_y, val_loss, W1, b1, W2, b2, W3, b3, val_size, val_acc)


'''
    Show the error rate
'''
x_axis = np.arange(1,len(train_loss)+1)
import matplotlib.pyplot as plt
plt.plot(x_axis*2, train_loss, 'r', label='Training Loss')
plt.legend()
plt.plot(x_axis*2, val_loss, 'g', label='Validation Loss')
plt.legend()
plt.xlabel('Training epoch')
plt.ylabel('Loss')
plt.show()
plt.close()

'''
    Show the accuracy rate
'''
x_axis = np.arange(1,len(train_acc)+1)
import matplotlib.pyplot as plt
plt.plot(x_axis*2, train_acc, 'r', label='Training Accuracy')
plt.legend()
plt.plot(x_axis*2, val_acc, 'g', label='Validation Accuracy')
plt.legend()
plt.xlabel('Training epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.show()

'''
    Show the error and accuracy for test
'''

test_size = test_X.shape[0]
test_loss, test_accuracy = test_loss_and_accuracy(test_X, test_y, W1, b1, W2, b2, W3, b3, test_size)
print(test_loss, test_accuracy)