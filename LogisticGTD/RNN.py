import tensorflow as tf
import DataPreper as dp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops import rnn, rnn_cell

def f(z):
    return 1 / (1.0 + tf.exp(-z))


train_x,train_y,test_x,test_y = dp.TrainAndTestRNN(97)
# number of features
n_featurs = train_x[0].size
# data size
total_size = train_y.size

#number of previous attacks we want to cheack before the current
batch_size = 8

hm_epochs = 512
#classes number
n_classes = 1

rnn_size = 256


# creates batches
curr_batch = 0
def next_batch(data_x,data_y):
    global curr_batch
    start = curr_batch * batch_size
    if start + batch_size >= data_y.size:
        return (None, None)
    end = start + batch_size
    batch_xs = []
    _x = []
    batch_ys = data_y[start:end]
    batch_ys = np.array(batch_ys)
    #make batc_xs as a 3D array
    for row in data_x[start:end]:
        _x.append(row)
    batch_xs.append(_x)

    batch_xs = np.array(batch_xs)

    curr_batch += 1
    return (batch_xs,batch_ys)

def recurrent_neural_network():
    #input layer
    x = tf.placeholder('float', [None,batch_size, n_featurs])
    y = tf.placeholder('float', [None, n_classes])

    #lstm cell
    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True,forget_bias=0.0)
    outputs, states = rnn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    outputs = tf.transpose(outputs,[1,0,2])
    #get the last outpus from the lstm
    last = outputs[-1]

    #one layer NN
    W = tf.Variable(tf.truncated_normal([rnn_size, n_classes], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[n_classes]))
    output = f(tf.matmul(last,W) + b)

    #TRAINING
    loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output), reduction_indices=[1]))

    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        global curr_batch
        axis_y = []
        axis_x = []
        for epoch in range(hm_epochs):
            epoch_loss = 0
            acc = 0
            curr_batch = 0

            while True:
                epoch_x,epoch_y = next_batch(train_x,train_y)
                if epoch_x is None:
                    break
                else:
                    _, c = sess.run([optimizer, loss], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c
                    acc += accuracy.eval(feed_dict={x: epoch_x, y: epoch_y})
                    axis_y.append(c)
                    axis_x.append(epoch)

            #print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss,'training accuracy ',acc/curr_batch)
        plt.ylabel('LOSS')
        plt.xlabel('EPHOCH')
        plt.title('Train Loss Session')
        plt.plot(axis_x, axis_y, 'b--')
            #TEST
        curr_batch = 0
        total = 0
        fneg = 0
        fpos = 0
        tneg = 0
        tpos = 0

        while True:
            epoch_x, epoch_y = next_batch(test_x, test_y)
            total += 1
            if epoch_x is None:
                break
            else:
                prediction = output.eval(session=sess, feed_dict = 	{x:epoch_x})
                #check equal to the next value
                # print(test_y[curr_batch+batch_size][0],prediction[0][0])
                total += 1
                # True Positive
                if (np.around(prediction[0][0]) == 1 and test_y[curr_batch + batch_size][0] == 1):
                    # print("SUCCESS! predict: ", h(test, w, b), " round: ", np.around(h(test, w, b)), " actual: ", test_y[i])
                    tpos += 1
                # False Negative
                elif (np.around(prediction[0][0]) == 0 and test_y[curr_batch + batch_size][0] == 1):
                    fneg += 1
                # False Positive
                elif (np.around(prediction[0][0]) == 1 and test_y[curr_batch + batch_size][0] == 0):
                    fpos += 1
                # True Negative
                else:
                    tneg += 1
        prec = tpos / (tpos + fpos)
        acc = tpos / total
        reca = tpos / (tpos + fneg)
        print("Accuracy: ", acc, "%")
        print("Recall: ", reca)
        print("Precision: ", prec)
        print("F-Measure: ", 2 * (prec * reca) / (prec + reca))
        plt.show()



recurrent_neural_network()


