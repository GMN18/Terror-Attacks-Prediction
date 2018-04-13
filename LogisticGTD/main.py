import models as mdl
import DataPreper as dp


def main_menu():
    dp.clear()
    #get information from the user
    print("Hello,\n")
    print("Please insert known information on the attack:\n         [property1=numVal1 property2=... e.g. nkill=2 nwound=3] ")
    info = input()
    infoarr = info.split(" ")
    cols = []
    infoVec = []
    for inf in infoarr:
        tup = inf.split("=")
        cols.append(tup[0])
        infoVec.append(tup[1])

    print("Please choose the property predict :[e.g. weaptype1, attacktype1]")
    pred = input("Property: ")
    print("Please insert number of categories of this property:")
    n_pred = input("num of categories: ")
    mdl.SoftMax(infoVec,cols,pred,int(n_pred))

    return


main_menu()