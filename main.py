import pandas as pd
import numpy as np
import data_transformation as dt
#import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('train.csv')
validation = pd.read_csv('test.csv')

def plot(X):
    for i in range(0, col):
        for j in range(i + 1, col):
            plt.scatter(X[:,i], data['Survived'], color='r')
            plt.scatter(X[:,j], data['Survived'], color='b')
            plt.show()
    for i in range(0, col):
        plt.hist(X[:,i])
        plt.show()

def write_csv(result, line):
    Id = []
    for i in range(0, np.shape(X_val)[0]):
        Id.append(i + line + 1)

    df = pd.DataFrame({'PassengerId': Id, 'Survived': result}, index=None)
    pd.DataFrame(df).to_csv('Result.csv', header=['PassengerId', 'Survived'],
            index=None)

def print_data(data, key):
    S = []
    D = []
    YS = []
    YD = []
    for i in range(0, len(data[key])):
        if (data['Survived'][i] == 1):
            S.append(data[key][i])
        else:
            D.append(data[key][i])
    plt.hist(S, alpha=0.5, label='Surived', color='r')
    plt.hist(D, alpha=0.5, label='Died',color='b')
    plt.xlabel(key)
    plt.legend(['Survived', 'Died'])
    plt.show()

def main_test(X, Y, data):
    line, col = np.shape(X)
#    for j in range(0, col):
#        print(X[0][j], j)
    X = dt.create_sex_feature(X, 1)
    X = dt.create_emb_feature(X, 6) 
    X = dt.create_cabin_feature(X, 5)
    X = dt.change_nan_class(X, Y)
    X = dt.modify_pclass(X, 0)
    X = dt.modify_parch(X, 4)
    X = dt.modify_age(X, 2)

    train_length = int(line * 0.80)
    test_length = line - train_length

    X_train = X[:train_length]
    Y_train = Y[:train_length]
    X_test = X[train_length:]
    Y_test = Y[train_length:]

    rf = RandomForestClassifier(n_estimators=128)
    rf.fit(X_train, Y_train)
    pred = rf.predict(X_test)

    a = 1
    for i in range(0, test_length):
        if (pred[i] == Y_test[np.shape(X_train)[0] + i]):
            a += 1
    print(a / test_length)

def main(X, Y, X_val):
    X = dt.create_sex_feature(X, 1)
    X = dt.create_emb_feature(X, 6) 
    X = dt.create_cabin_feature(X, 5)
    X = dt.change_nan_class(X, Y)
    X = dt.modify_pclass(X, 0)
    X = dt.modify_parch(X, 4)
    X = dt.modify_age(X, 2)

    X_val = dt.create_sex_feature(X_val, 1)
    X_val = dt.create_emb_feature(X_val, 6) 
    X_val = dt.create_cabin_feature(X_val, 5)
    X_val = dt.change_nan_class(X_val, Y)
    X_val = dt.modify_pclass(X_val, 0)
    X_val = dt.modify_parch(X_val, 4)
    X_val = dt.modify_age(X_val, 2)

    rf = RandomForestClassifier(n_estimators=1280)
    rf.fit(X, Y)

    result = rf.predict(X_val)
    write_csv(result, np.shape(X)[0])

if __name__ == '__main__':
    drop = ['Survived', 'Ticket', 'Name', 'PassengerId', 'Fare']
    drop_val = ['Ticket', 'Name', 'PassengerId', 'Fare']
    X = np.array(data.drop(drop, axis=1).values)
    X_val = np.array(validation.drop(drop_val, axis=1).values)
    Y = data['Survived']
    main(X, Y, X_val)
    #main_test(X, Y, data)
