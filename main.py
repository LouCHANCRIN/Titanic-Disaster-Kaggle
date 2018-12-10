import pandas as pd
import numpy as np
import data_transformation as dt
#import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('train.csv')
validation = pd.read_csv('test.csv')

drop = ['Survived', 'Cabin', 'Embarked', 'Name',
        'Ticket', 'Fare', 'Cabin']

drop_sex = ['Survived', 'Cabin', 'Embarked', 'Name', 'Sex',
        'Ticket', 'Fare', 'Cabin']

drop_val = ['Cabin', 'Embarked', 'Name',
        'Ticket', 'Fare', 'Cabin']

X = [np.insert(row, 0, 1) for row in data.drop(drop, axis=1).values]
line, col = np.shape(X)
X = np.reshape(X, (line, col))
Y = data['Survived']
X_val = [np.insert(row, 0, 1) for row in validation.drop(drop_val, axis=1).values]

def plot(X):
    for i in range(0, col):
        for j in range(i + 1, col):
            plt.scatter(X[:,i], data['Survived'], color='r')
            plt.scatter(X[:,j], data['Survived'], color='b')
            plt.show()
    for i in range(0, col):
        plt.hist(X[:,i])
        plt.show()

def write_csv(result):
    Id = []
    for i in range(0, np.shape(X_val)[0]):
        Id.append(i + line + 1)

    df = pd.DataFrame({'PassengerId': Id, 'Survived': result}, index=None)
    pd.DataFrame(df).to_csv('Result.csv', header=['PassengerId', 'Survived'],
            index=None)

def main_test(X, Y):
    X = dt.create_sex_feature(X, 3)
    X = dt.change_nan_class(X, Y)

    train_length = int(line * 0.85)
    test_length = line - train_length

    X_train = X[:train_length]
    Y_train = Y[:train_length]
    X_test = X[train_length:]
    Y_test = Y[train_length:]

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, Y_train)
    pred = rf.predict(X_test)

    a = 1
    for i in range(0, test_length):
        if (pred[i] == Y_test[np.shape(X_train)[0] + i]):
            a += 1
    print(a / test_length)

def main(X, Y, X_val):
    X = dt.create_sex_feature(X, 3)
    X = dt.change_nan_class(X, Y)
    X_val = dt.create_sex_feature(X_val, 3)
    X_val = dt.change_nan(X_val)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, Y)

    result = rf.predict(X_val)
    write_csv(result)

if __name__ == '__main__':
   #main(X, Y, X_val)
   main_test(X, Y)
