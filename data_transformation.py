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

#def modify_fare(X, j):
#    line, col = np.shape(X)
#    for i in range(0, line):
#        if (X[i][j] < 20):
#            X[i][j] = -1
#        else:
#            X[i][j] = 1
#    return (X)

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

#def get_number_cabin(cab):
#   i = 1
#    while (i < len(cab) - 1 and cab[i] >= '0' and cab[i] <= '9'):
#        i += 1
#    if (i > 1):
#        return (int(cab[1:i + 1]))
#    return (0)

#def create_cabin_feature(X, j):
#    line, col = np.shape(X)
#    number_cabin = [0] * line
#    for i in range(0, line):
#        if (X[i][j] == X[i][j]):
#            number_cabin[i] = get_number_cabin(X[i][j])
#            if (X[i][j][0] == 'A'):
#                X[i][j] = 0
#            elif (X[i][j][0] == 'B'):
#                X[i][j] = 1
#            elif (X[i][j][0] == 'C'):
#                X[i][j] = 2
#            elif (X[i][j][0] == 'D'):
#                X[i][j] = 3
#            elif (X[i][j][0] == 'E'):
#                X[i][j] = 4
#            elif (X[i][j][0] == 'F'):
#                X[i][j] = 5
#            elif (X[i][j][0] == 'G'):
#                X[i][j] = 6
#            elif (X[i][j][0] == 'T'):
#                X[i][j] = 7
#        else:
#            X[i][j] = 8
#    return (np.append(X, np.reshape(number_cabin, (line, 1)), axis=1))

#def get_surname(X, j, c):
#    line, col = np.shape(X)
#    surname = []
#    for i in range(0, line):
#        for k in range(0, len(X[i][j])):
#            if (X[i][j][k] == c):
#                surname.append(X[i][j][:k])
#    return (surname)

#def check_sibsp_parch(data, i, j):
#    if (((data['SibSp'][i] > 0 and data['SibSp'][j] > 0)
#            or (data['Parch'][i] > 0 and data['Parch'][j] > 0))
#            and data['Embarked'][i] == data['Embarked'][j]):
#        return (1)
#    return(0)

#def check_if_same_family(X, data, dic, key, i, j):
#    a = 0
#    b = 0
#    c = 0
#    if (check_sibsp_parch(data, dic[key][i], dic[key][j]) == 1):
#        if (data['Cabin'][dic[key][i]] == data['Cabin'][dic[key][i]]
#                and data['Cabin'][dic[key][j]] != data['Cabin'][dic[key][j]]):
#            a = 1
#            b = i
#            c = j
#        if (data['Cabin'][dic[key][j]] == data['Cabin'][dic[key][j]]
#                and data['Cabin'][dic[key][i]] != data['Cabin'][dic[key][i]]):
#            a = 1
#            b = j
#            c = i
#    return (a, b, c)

#def create_name_feature(X, name_col, Y, data):
#    line, col = np.shape(X)
#    surname = get_surname(X, name_col, ',')
#    dic = {}
#    if (np.shape(surname)[0] != line):
#        print("Error, number of passengers found doesn't match number of passengers in the set")
#        print(line, "passengers in the set,", np.shape(surname)[0], "passengers found")
#    for i in range(0, line):
#        if (surname[i] in dic):
#            dic[surname[i]].append(i)
#        else:
#            dic[surname[i]] = [i]
#    x = 0
#    for key in dic:
#        if (len(dic[key]) > 1):
#            for i in range(0, len(dic[key])):
#                for j in range(i + 1, len(dic[key])):
#                    a, b, c = check_if_same_family(X, data, dic, key, i, j)
#                    if (a == 1):
#                        print("changed nan")
#                        X[b][7] = X[c][7]
#    for i in range(0, line):
#        X[i][name_col] = 0
#    return (X)
