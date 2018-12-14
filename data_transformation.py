import numpy as np

def moyenne_class(x, y, i):
    som0 = 0
    som1 = 0
    count0 = 0
    count1 = 0
    line, col = np.shape(x)
    for j in range(0, line):
        if (x[j][i] == x[j][i]):
            if (y[j] == 0):
                som0 += x[j][i]
                count0 += 1
            else:
                som1 += x[j][i]
                count1 += 1
    return (som0 / count0, som1 / count1)

def change_nan_class(X, Y):
    a = 0
    line, col = np.shape(X)
    for i in range(0, col):
        moy0, moy1 = moyenne_class(X, Y, i)
        for j in range(0, line):
            a += 1
            if (X[j][i] != X[j][i]):
                if (Y[j] == 0):
                    X[j][i] = moy0
                else:
                    X[j][i] = moy1
    X = np.reshape(X, (line, col))
    return (X)

def moyenne(x, j):
    som = 0
    count = 0
    line, col = np.shape(x)
    for i in range(0, line):
        if (x[i][j] == x[i][j]):
            som += x[i][j]
            count += 1
    return (som / count)

def change_nan(X):
    a = 0
    line, col = np.shape(X)
    for i in range(0, col):
        moy = moyenne(X, i)
        for j in range(0, line):
            a += 1
            if (X[j][i] != X[j][i]):
                X[j][i] = moy
    X = np.reshape(X, (line, col))
    return (X)

def create_sex_feature(X, j):
    line, col = np.shape(X)
    for i in range(0, line):
        if (X[i][j] == 'female'):
            X[i][j] = 1
        else:
            X[i][j] = -1
    return (X)

def modify_pclass(X, j):
    line, col = np.shape(X)
    for i in range(0, line):
        if (X[i][j] == 2):
            X[i][j] = 0
        elif (X[i][j] == 3):
            X[i][j] = -1
    return (X)

def modify_age(X, j):
    line, col = np.shape(X)
    for i in range(0, line):
        if (X[i][j] <= 10):
            X[i][j] = 2
        elif (X[i][j] <= 20):
            X[i][j] = 1
        else:
            X[i][j] = 0
    return (X)

def modify_parch(X, j):
    line, col = np.shape(X)
    for i in range(0, line):
        if (X[i][j] == 0):
            X[i][j] = -1
        elif (X[i][j] > 1):
            X[i][j] = 1
        else:
            X[i][j] = 0
    return (X)

def modify_sibsp(X, j):
    line, col = np.shape(X)
    for i in range(0, line):
        if (X[i][j] == 1 or X[i][j] == 2):
            X[i][j] = 1
        else:
            X[i][j] = -1
    return (X)

def create_cabin_feature(X, j):
    line, col = np.shape(X)
    for i in range(0, line):
        if (X[i][j] == X[i][j]):
            X[i][j] = 1
        else:
            X[i][j] = 0
    return (X)

def create_emb_feature(X, j):
    line, col = np.shape(X)
    for i in range(0, line):
        if (X[i][j] != X[i][j] or X[i][j] == 'C'):
            X[i][j] = 1
        elif (X[i][j] == 'S'):
            X[i][j] = -1
        elif (X[i][j] == 'Q'):
            X[i][j] = 1
    return (X)
