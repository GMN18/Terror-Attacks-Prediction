#!/usr/bin/python
import numpy as np
import DataPreper as dp
import tensorflow as tf
from fractions import _gcd
from tensorflow.contrib import rnn
from tensorflow.python.ops import rnn, rnn_cell


train_x,train_y,test_x,test_y = dp.TrainAndTestRNN()
# number of features
n_featurs = train_x[0].size
# data size
total_size = train_y.size

batch_size = _gcd(train_y.size,test_y.size)



    # gets the data after the split
    # the X data contains stock prices of 19 consecutive days
    # the y data is the stock price of the 20th day
    # reshape to a 3-dimensional tensor

#train_x = train_x.reshape(train_x.shape + tuple([1]))
#test_x = test_x.reshape(test_x.shape + tuple([1]))
#train_y = train_y.reshape(train_y.shape + tuple([1]))
#test_y = test_y.reshape(test_y.shape + tuple([1]))


# rnn configuration
cellsize = 30
output_size = 1
epochs = 180
n_classes = 2

# input
x = tf.placeholder(tf.float32, shape=[batch_size,n_featurs])
y = tf.placeholder(tf.float32, shape=[batch_size, output_size])

def recurrent_neural_network():
    layer = {'weights':tf.Variable(tf.random_normal([n_featurs,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(cellsize, forget_bias=0.0)
    output, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.transpose(output, [1, 0, 2])
    last = output[-1]

    W = tf.Variable(tf.truncated_normal([n_featurs, n_classes], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[n_classes]))

    z = tf.matmul(last, W) + b
    res = tf.nn.softmax(z)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(res), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(res, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for ephoch in range(epochs):
        acc = 0
        for curr_batch in range(0, train_y.size/batch_size):
                # creates batches
                 start = curr_batch * batch_size
                 end = start + batch_size
                 batch_xs = train_x[start:end]
                 batch_ys = train_y[start:end]
                #Training
                 sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
                 acc += accuracy.eval(feed_dict={x: batch_xs, y: batch_ys})

                 print("step %d, training accuracy %g" % (ephoch, acc / curr_batch))

        #lstm_cell = rnn_cell.BasicLSTMCell(cell_size,state_is_tuple=True)
    #outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    #output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']
    #y_ = f(output)
    #return output


recurrent_neural_network()

