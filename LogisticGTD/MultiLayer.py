import tensorflow as tf
import numpy as np
import DataPreper as dp

def f(z):
    return 1 / (1.0 + tf.exp(-z))

train_data = dp.getTrainNN()
train_x = train_data[0]
train_y = train_data[1]

test_data = dp.getTestNN()
test_x = test_data[0]
test_y = test_data[1]

best = 0
bestS = 0
features = 6
i = 100
while i<10000:

    (layer1,layer2,layer3) = (i,1,25)
    x = tf.placeholder(tf.float32,[None,features])
    y_ = tf.placeholder(tf.float32,[None,1])
    #Hidden Layer 1; size 100 ; input layer
    W1 = tf.Variable(tf.truncated_normal([features,layer1], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[layer1]))
    z1 = tf.add(tf.matmul(x,W1),b1)
    a1 = tf.nn.relu(z1)
    #Hidden Layer 2; size 50
    W2 = tf.Variable(tf.truncated_normal([layer1,layer2], stddev=0.1))
    b2 = tf.Variable(0.)
    z2 = tf.matmul(a1,W2) + b2
    #a2 = tf.nn.relu(z2)
    #Layer 3; size 25;
    #W3 = tf.Variable(tf.truncated_normal([layer2,layer3],stddev=0.1))
    #b3 = tf.Variable(tf.constant(0.1, shape=[layer3]))
    #z3 = tf.matmul(a2,W3) + b3
    #a3 = tf.nn.relu(z3)
    #Layer 4; size 1; output layer
    #W4 = tf.Variable(tf.truncated_normal([layer3,1],stddev=0.1))
    #b4 = tf.Variable(0.)
    #z4 = tf.matmul(a3,W4) + b4
    y = f(z1)

    loss = tf.reduce_mean(-(y_*tf.log(y)+(1-y_)*tf.log(1-y)))
    update = tf.train.AdamOptimizer(0.1).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    losses = 0
    for j in range(50000):
            __,c = sess.run([update,loss], feed_dict = {x:train_x, y_:train_y})
            losses += c
            if j < 2:
                print("loss: ",__)
            #print("W1:\n",W1)
            #print("W2:\n",W2)
    #print('prediction: ', y.eval(session=sess, feed_dict = 	{x:test_x}))
    predictionList = y.eval(session=sess, feed_dict = 	{x:test_x})
    counter = 0
    success = 0
    for k,num in enumerate(predictionList):
        counter += 1
        if (np.around(num[0]) == test_y[k][0]):
            #print("SUCCESS! predict: ",num[0] , " round: ", np.around(num[0]), " actual: ", test_y[i][0])
            success=success+1
        #else:
            #print("FAIL! predict: ", num[0], " round: ", np.around(num[0]), " actual: ", test_y[i][0])

    print("nodes at layer 1",i," num of predict success: ",success)
    if(success > best):
        best = i
        bestS = success
    i=i+100

print("best num of nodes: ",best," with success of ",bestS)