import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler


def rmse_metric(actual, predicted):
    sum_err = 0.0
    for i in range(len(actual)):
        err = predicted[i] - actual[i]
        sum_err += err ** 2
    mean_err = sum_err / (len(actual)-1)
    return np.sqrt(mean_err)


if __name__ == '__main__':
    dataset = np.array(pd.read_csv("winequality-white.csv", sep=',', header=None))
    k_Cross = KFold(n_splits=10, random_state=0, shuffle=True)
    index = 0
    score = np.array([])
    Scaler = MinMaxScaler()
    data,label = dataset[:,:-1],dataset[:,-1]
    data = Scaler.fit_transform(data)
    for train_index, test_index in k_Cross.split(dataset):
        train_data, train_label = data[train_index, :], label[train_index]
        test_data, test_label = data[test_index, :], label[test_index]
        model = SGDRegressor()
        model.fit(train_data, train_label)
        pred = model.predict(test_data)
        rmse = rmse_metric(test_label, pred)
        score = np.append(score,rmse)
        print('score[{}] = {}'.format(index,rmse))
        index+=1
    print('mean_rmse = {}'.format(np.mean(score)))