import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


def rmse_metric(actual, predicted):
    sum_err = 0.0
    for i in range(len(actual)):
        err = predicted[i] - actual[i]
        sum_err += err ** 2
    mean_err = sum_err / (len(actual)-1)
    return np.sqrt(mean_err)


if __name__ == '__main__':
    dataset = np.array(pd.read_csv("abalone.csv", sep=',', header=None))
    k_Cross = KFold(n_splits=5, random_state=0, shuffle=True)
    index = 0
    scores = np.array([])
    Scaler = MinMaxScaler()
    data,label = dataset[:,:-1],dataset[:,-1]
    data = Scaler.fit_transform(data)
    for train_index, test_index in k_Cross.split(dataset):
        train_data, train_label = data[train_index, :], label[train_index]
        test_data, test_label = data[test_index, :], label[test_index]
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(train_data, train_label)
        pred = model.predict(test_data)
        score = rmse_metric(test_label, pred)
        scores = np.append(scores,score)
        print('score[{}] = {}'.format(index,score))
        index+=1
    print('mean_rmse = {}'.format(np.mean(scores)))