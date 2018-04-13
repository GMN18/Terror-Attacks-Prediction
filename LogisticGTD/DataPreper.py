import numpy as np
import csv
import sys, os


# define the clear screan function
def get_platform():
    platforms = {
        'linux1': 'Linux',
        'linux2': 'Linux',
        'darwin': 'OS X',
        'win32': 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform

    return platforms[sys.platform]
#clear the console
def clear():
	if(get_platform()=='Linux'):
		os.system('clear')
	else:
		os.system('cls')

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i

#this function get as input an array of columns tot take from the Data Base and return Train x and y and Test x and y
# and get the wanted SoftMax column
#if no input the default values are ['country','region','targtype1','targsubtype1','nkill','nwound'] and 'weaptype'
def getTrainAndTestSoftMax(columns = ['country','region','targtype1','nkill','nwound'],SoftMaxVal = 'weaptype1',SoftMaxLen = 13,fileName = 'globalterrorismdb_dist.csv'):
    test_x = []
    test_y = []
    train_y = []
    train_x = []
    SoftMaxSet = []
    print('START loading data...')
    numOfLines = file_len(fileName)

    with open(fileName) as csvfile:

        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            #print(int(i/numOfLines*100),'% Loading data...\r')
            #init the one hot vector
            one_hot_softmax = [0 for i in range(SoftMaxLen)]
            if(row[SoftMaxVal]!=''):
                one_hot_softmax[int(row[SoftMaxVal])-1] = 1
                if (int(row[SoftMaxVal]),row[SoftMaxVal+"_txt"]) not in SoftMaxSet:
                    SoftMaxSet.append((int(row[SoftMaxVal]),row[SoftMaxVal+"_txt"]))

            #perppering the data
            data = []
            for col in columns:
                if(row[col]==''):
                    data.append(0)
                else:
                    data.append(row[col])

            #separte for Train and Test
            if(i < 0.7*numOfLines):
                train_y.append(one_hot_softmax)
                train_x.append(data)
            else:
                test_y.append(one_hot_softmax)
                test_x.append(data)

    print('DONE loading data...')
    return test_x,test_y,train_x,train_y,SoftMaxSet



def readCSVtrain():
    arr = []
    with open('GTDLogistic2.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for i,row in enumerate(reader):
            if(i==5000):
                break
            x = []
            x.append(row['weaptype(hot = 1)'])
            #x.append(row['country'])
            x.append(row['region'])
            #x.append(row['targtype1'])
            #x.append(row['targsubtype1'])
            x.append(row['nkill'])
            x.append(row['nwound'])
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
                x.append(row['region'])
                x.append(row['targtype1'])
                x.append(row['targsubtype1'])
                x.append(row['nkill'])
                x.append(row['nwound'])
                arr.append(x)
    return arr

def getTrain():
    data_x = readCSVtrain()

    #cast the strings to float
    for i in range(len(data_x)):
        for j in range(len(data_x[i])):
            if(data_x[i][j]==''):
                data_x[i][j]=0.
            else:
                data_x[i][j]=float(data_x[i][j])

    #TRAIN
    train_x = np.full((len(data_x),6),0,dtype=np.longdouble)
    train_y = np.empty(len(data_x),dtype=np.longdouble)


    counter = 0
    for j,row in enumerate(data_x):
        train_x[j] = np.array([row[1],row[2],row[3],row[4],row[5],row[6]])
        train_y[j] = row[0]


    #cast the strings to float
    for i in range(len(train_x)):
        for j in range(len(train_x[i])):
            if(train_x[i][j]==''):
                train_x[i][j]=0.
            else:
                train_x[i][j]=float(train_x[i][j])



    for i in range(len(train_y)):
        train_y[i] = float(train_y[i])

    return [train_x,train_y]

def getTest():
    #TEST
    data_x_test = readCSVtest()
    #set training 70% and test 30%
    test_x = np.full((len(data_x_test),6),0,dtype=np.longdouble)
    test_y = np.empty(len(data_x_test),dtype=np.longdouble)

    for j,row in enumerate(data_x_test):
        for i,x in enumerate(row):
            if(row[i]==''):
                row[i] = 0

        test_x[j] = np.array([row[1],row[2],row[3],row[4],row[5],row[6]])
        test_y[j] = np.array(row[0])


    #cast the strings to float
    for i in range(len(test_x)):
        for j in range(len(test_x[i])):
            if(test_x[i][j]==''):
                test_x[i][j]=0.
            else:
                test_x[i][j]=float(test_x[i][j])


    for i in range(len(test_x)):
        test_y[i] = float(test_y[i])

    return [test_x,test_y]

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
            x.append(row['region'])
            x.append(row['targtype1'])
            x.append(row['targsubtype1'])
            x.append(row['nkill'])
            x.append(row['nwound'])
            arr.append(x)
    return arr

def getTrainNN():
    data_x = readCSVtrain()

    #cast the strings to float
    for i in range(len(data_x)):
        for j in range(len(data_x[i])):
            if(data_x[i][j]==''):
                data_x[i][j]=0.
            else:
                data_x[i][j]=float(data_x[i][j])

    #TRAIN
    train_x = np.full((len(data_x),6),0,dtype=np.longdouble)
    train_y = np.empty((len(data_x),1),dtype=np.longdouble)


    counter = 0
    for j,row in enumerate(data_x):
        train_x[j] = np.array([row[1],row[2],row[3],row[4],row[5],row[6]])
        train_y[j][0] = row[0]


    #cast the strings to float
    for i in range(len(train_x)):
        for j in range(len(train_x[i])):
            if(train_x[i][j]==''):
                train_x[i][j]=0.
            else:
                train_x[i][j]=float(train_x[i][j])



    for i in range(len(train_y)):
        train_y[i] = float(train_y[i])

    return [train_x,train_y]

def getTestNN():
    #TEST
    data_x_test = readCSVtest()
    #set training 70% and test 30%
    test_x = np.full((len(data_x_test),6),0,dtype=np.longdouble)
    test_y = np.empty((len(data_x_test),1),dtype=np.longdouble)

    for j,row in enumerate(data_x_test):
        for i,x in enumerate(row):
            if(row[i]==''):
                row[i] = 0

        test_x[j] = np.array([row[1],row[2],row[3],row[4],row[5],row[6]])
        test_y[j][0] = row[0]


    #cast the strings to float
    for i in range(len(test_x)):
        for j in range(len(test_x[i])):
            if(test_x[i][j]==''):
                test_x[i][j]=0.
            else:
                test_x[i][j]=float(test_x[i][j])


    for i in range(len(test_x)):
        test_y[i] = float(test_y[i])

    return [test_x,test_y]

def getFloat(str):
    if(str!=''):
        return int(str)
    else:
        return 0

def readCSVRNN(countryNumber = 217):
    arrX = []
    arrY = []
    with open('GTDLogistic2.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for i,row in enumerate(reader):
            if int(row['country']) != countryNumber:
                y = []
                x = []
                y.append(getFloat(row['weaptype(hot = 1)']))
                x.append(getFloat(row['country']))
                x.append(getFloat(row['region']))
                x.append(getFloat(row['targtype1']))
                x.append(getFloat(row['targsubtype1']))
                x.append(getFloat(row['nkill']))
                x.append(getFloat(row['nwound']))
                x = np.array(x)
                y = np.array(y)
                arrX.append(x)
                arrY.append(y)
    arrX = np.array(arrX)
    arrY = np.array(arrY)
    return arrX,arrY

def TrainAndTestRNN(countryNumer = 217):
    data_x,data_y = readCSVRNN()
    testFinalIndex = int(data_y.size*0.7)
    train_x = data_x[:testFinalIndex]
    train_y = data_y[:testFinalIndex]
    test_x = data_x[testFinalIndex:]
    test_y = data_y[testFinalIndex:]

    return train_x,train_y,test_x,test_y
