import tensorflow as tf
import numpy as np
import DataPreper as dp
import matplotlib.pyplot as plt

def f(z):
    return 1 / (1.0 + tf.exp(-z))

train_data = dp.getTrainNN()
train_x = train_data[0]
train_y = train_data[1]

test_data = dp.getTestNN()
test_x = test_data[0]
test_y = test_data[1]
features = 6
layer_node_num = 128

x = tf.placeholder(tf.float32,[None,features])
y_ = tf.placeholder(tf.float32,[None,1])
#Hidden Layer 1; size 100 ; input layer
W1 = tf.Variable(tf.truncated_normal([features,layer_node_num], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[layer_node_num]))
z1 = tf.add(tf.matmul(x,W1),b1)
a1 = tf.nn.relu(z1)
#Hidden Layer 2;
W2 = tf.Variable(tf.truncated_normal([layer_node_num,1], stddev=0.1))
b2 = tf.Variable(0.)
z2 = tf.matmul(a1,W2) + b2
#a2 = tf.nn.relu(z2)
#Layer 3;
#W3 = tf.Variable(tf.truncated_normal([layer2,layer3],stddev=0.1))
#b3 = tf.Variable(tf.constant(0.1, shape=[layer3]))
#z3 = tf.matmul(a2,W3) + b3
#a3 = tf.nn.relu(z3)
#Layer 4;
#W4 = tf.Variable(tf.truncated_normal([layer3,1],stddev=0.1))
#b4 = tf.Variable(0.)
#z4 = tf.matmul(a3,W4) + b4
#output layer
y = f(z2)

loss = tf.reduce_mean(-(y_*tf.log(y)+(1-y_)*tf.log(1-y)))
update = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
axis_y = []
axis_x = []
for j in range(0,10000):
    __,currentLoss = sess.run([update,loss], feed_dict = {x:train_x, y_:train_y})
    if (j % 10 == 0):
        # print("current loss at epoch ",i,": ",currentLoss)
        axis_y.append(currentLoss)
        axis_x.append(j)

plt.ylabel('LOSS')
plt.xlabel('EPHOCH')
plt.title('Train Loss Session')
plt.plot(axis_x, axis_y, 'b--')

#print('prediction: ', y.eval(session=sess, feed_dict = 	{x:test_x}))
predictionList = y.eval(session=sess, feed_dict = 	{x:test_x})
total = 0
fneg = 0
fpos = 0
tneg = 0
tpos = 0
#Test
for i,num in enumerate(predictionList):
    total += 1
    #True Positive
    if(np.around(num[0])==1 and test_y[i][0]==1):
        #print("SUCCESS! predict: ", h(test, w, b), " round: ", np.around(h(test, w, b)), " actual: ", test_y[i])
        tpos += 1
    #False Negative
    elif(np.around(num[0])==0 and test_y[i][0]==1):
        fneg+=1
    #False Positive
    elif(np.around(num[0])==1 and test_y[i][0]==0):
        fpos+=1
    #True Negative
    else:
        tneg+=1

prec = tpos/(tpos+fpos)
acc = tpos/total
reca = tpos/(tpos+fneg)
print("Accuracy: ",acc,"%")
print("Recall: ",reca)
print("Precision: ",prec)
print("F-Measure: ",2*(prec*reca)/(prec+reca))
plt.show()