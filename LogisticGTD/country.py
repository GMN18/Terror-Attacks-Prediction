import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

def h(x,w,b):
    return 1 / (1+np.exp(-(np.dot(x,w) + b)))

def readCSVtrain():
    arr = []
    with open('GTDLogistic2.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for i,row in enumerate(reader):
            if(i==5000):
                break
            x = []
            x.append(row['weaptype(hot = 1)'])
            x.append(row['country'])
            arr.append(x)
    return arr

def readCSVtest():
    arr = []
    with open('GTDLogistic2.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for i,row in enumerate(reader):
            if(i>5000):
                x = []
                x.append(row['weaptype(hot = 1)'])
                x.append(row['country'])
                arr.append(x)
    return arr


data_x = readCSVtrain()

#cast the strings to float
for i in range(len(data_x)):
    for j in range(len(data_x[i])):
        if(data_x[i][j]==''):
            data_x[i][j]=0.
        else:
            data_x[i][j]=float(data_x[i][j])

#TRAIN
train_x = np.full(len(data_x),0,dtype=np.longdouble)
train_y = np.empty(len(data_x),dtype=np.longdouble)


counter = 0
for j,row in enumerate(data_x):
    train_x[j] = row[1]
    train_y[j] = row[0]


#cast the strings to float
for i in range(len(train_x)):
        if(train_x[i]==''):
           train_x[i]=0.
        else:
            train_x[i]=float(train_x[i])



for i in range(len(train_y)):
    train_y[i] = float(train_y[i])

#TEST
data_x_test = readCSVtest()
#set training 70% and test 30%
test_x = np.full(len(data_x_test),0,dtype=np.longdouble)
test_y = np.empty(len(data_x_test),dtype=np.longdouble)

for j,row in enumerate(data_x_test):
    for i,x in enumerate(row):
        if(row[i]==''):
            row[i] = 0

    test_x[j] = row[1]
    test_y[j] = row[0]


#cast the strings to float
for i in range(len(test_x)):
        if(test_x[i]==''):
            test_x[i]=0.
        else:
            test_x[i]=float(test_x[i])


for i in range(len(test_x)):
    test_y[i] = float(test_y[i])

w = 0.
b = 0
alpha = 0.001

for iteration in range(100000):
    gradient_b = np.mean(1*(train_y-(h(train_x,w,b))))
    gradient_w = np.dot((train_y-h(train_x,w,b)), train_x)*1/len(train_y)
    b += alpha*gradient_b
    w += alpha*gradient_w

#Test
for i,test in enumerate(test_x):
    if(test>250):
            print("nwound: ",test," --> Prob of Firearms attack: ",h(test,w,b)," actual is: ",test_y[i])

#Test
y_axis = np.arange(0,1004,1)
plt.plot(y_axis,h(y_axis,w,b),'r--')
plt.ylabel('Country Code')
plt.xlabel('Prob For Firearms Attack')
plt.show()



