import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import DataPreper as dp

################################################
######      HELP FUNCTIONS       ###############
################################################

def findMaxIndex(vector):
    max = 0
    index = 0
    for i in range(len(vector)):
        if(vector[i]>max):
            max = vector[i]
            index = i

    return index

def findValuForIndex(index,TupSet):
    for num,val in TupSet:
        if num==index:
            return val

######################################################
#                SOFT MAX MODEL
######################################################
#SoftMax Session get as input the known info from the user and the wanted porperty to predict and the num of categories in this property
def SoftMax(infoVec,cols,pred,n_pred):
    #get the data
    train_x,train_y,test_x,test_y,SoftMaxSet = dp.getTrainAndTestSoftMax(cols,pred,n_pred,'globalterrorismdb_dist.csv')
    SoftMaxSet.sort()

    print(SoftMaxSet)

    #init nub of features and num of categorites
    numOfFeatures = len(train_x[0])
    numOfCategories = len(train_y[0])

    print("Num Of Features ",numOfFeatures," Num Of Categories ",numOfCategories)
    #input layer
    x = tf.placeholder(tf.float32, [None, numOfFeatures])
    y_ = tf.placeholder(tf.float32, [None, numOfCategories])

    #Wegiths and bias
    W = tf.Variable(tf.zeros([numOfFeatures,numOfCategories]))
    b = tf.Variable(tf.zeros([numOfCategories]))

    #output layer
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    #loss and update functions
    loss = -tf.reduce_mean(y_*tf.log(y))
    update = tf.train.GradientDescentOptimizer(0.000001).minimize(loss)

    #TRAINING
    print(">>>START TRAINING<<<<")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    axis_y = []
    axis_x = []
    #correct = tf.equal(findMaxIndex(y),findMaxIndex(y_))
    #accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    for i in range(0,1000):
        __,currentLoss = sess.run([update,loss], feed_dict = {x:train_x, y_:train_y})
        if(i%10==0):
            #print("current loss at epoch ",i,": ",currentLoss)
            axis_y.append(currentLoss)
            axis_x.append(i)

    plt.ylabel('LOSS')
    plt.xlabel('EPHOCH')
    plt.title('Train Loss Session')
    plt.plot(axis_x,axis_y,'b--')


    print(">>>DONE TRAINING<<<<")

    #TEST
    print("@@@START TEST@@@")

    predictionList = y.eval(session=sess, feed_dict = {x:test_x})
    total = len(predictionList)
    success = 0
    #print("len(predictionList) ",len(predictionList)," len(test_y) ",len(test_y))
    for k,vec in enumerate(predictionList):
            index = findMaxIndex(test_y[k])
            predIndex= findMaxIndex(vec)

            if(predIndex==index):
                success+=1
                #print("----SUCCESS-----\nPrediction ", predictionList[k])
                #print("Original ", test_y[k])
    print("@@@DONE TEST@@@")

    print("With Test Accuracy of ",int(success/total*100.0),"%")

    feed = [infoVec]
    predict = y.eval(session=sess, feed_dict = {x:feed})
    for res in predict:
        for i,val in enumerate(res):
            print("     Probabilty for ",findValuForIndex(i+1,SoftMaxSet),pred," is ",val)

    plt.show()
