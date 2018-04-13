import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import DataPreper as dp

def h(x,w,b):
    return 1 / (1+np.exp(-(np.dot(x,w) + b)))

train_x = dp.getTrain()[0]
train_y = dp.getTrain()[1]

test_x = dp.getTest()[0]
test_y = dp.getTest()[1]

w = np.array([0.,0,0,0,0,0])
b = 0
alpha = 0.001

for iteration in range(1000):
    gradient_b = np.mean(1*(train_y-(h(train_x,w,b))))
    gradient_w = np.dot((train_y-h(train_x,w,b)), train_x)*1/len(train_y)
    b += alpha*gradient_b
    w += alpha*gradient_w

total = 0
fneg = 0
fpos = 0
tneg = 0
tpos = 0
#Test
for i,test in enumerate(test_x):
    total += 1
    #print("in country: ",test[0],"region: ",test[1],"target type: ",test[2],"nkills: ",test[3]," nwound: ",test[4]," --> Prob of Firearms attack: ",h(test,w,b)," actual is: ",test_y[i])
    #True Positive
    if(np.around(h(test,w,b))==1 and test_y[i]==1):
        #print("SUCCESS! predict: ", h(test, w, b), " round: ", np.around(h(test, w, b)), " actual: ", test_y[i])
        tpos += 1
    #False Negative
    elif(np.around(h(test,w,b))==0 and test_y[i]==1):
        fneg+=1
    #False Positive
    elif(np.around(h(test,w,b))==1 and test_y[i]==0):
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


