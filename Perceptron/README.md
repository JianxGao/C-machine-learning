## 3.4 Perceptron

> 感知器算法是一种最简单的人工神经网络。它是一个单神经元模型，可用于两类分类问题，并为以后开发更大的网络提供了基础。在本教程中，您将了解如何使用C语言从头开始实现感知器算法。

### 3.4.1 算法介绍

#### 感知算法

感知器的灵感来自于单个神经细胞的信息处理过程，这种神经细胞被称为神经元。神经元通过树突接收输入信号，然后树突将电信号传递到细胞体。以类似的方式，感知器从训练数据的例子中接收输入信号，我们将其加权并组合成一个线性方程，这个方程被称为激活函数。

$$
activation = bias + \sum_{i=1}^{n}weight_i×x_i\tag{1.1}
$$

然后利用传递函数(如阶跃传递函数)将激活函数转化为输出值或预测值。

$$
prediction = 1.0 IF activation ≥ 0.0 ELSE 0.0\tag{1.2}
$$

这样，感知器就是一个解决两类问题（0和1）的分类算法，一个线性方程(如线或超平面)可以用来分离这两个类。它与线性回归和逻辑回归密切相关，后者以类似的方式进行预测(例如输入加权和)。感知器算法的权值必须使用随机梯度下降从训练数据中估计出来。

#### 随机梯度下降

感知器算法使用梯度下降来更新权重。梯度下降在多元线性回归一节中进行了介绍和描述。每次梯度下降迭代，权值w根据公式更新如下:

$$
w = w + learning\space rate × (expected-predicted) × x\tag{1.3}
$$

其中，w是被优化的权重，learning rate是一个需要你配置的学习率（例如0.01），

(如0.01)，(expected - predicted)为模型对训练数据归属权值的预测误差，x为输入值。(expected - predicted)为模型对带有权值的训练数据的预测误差，x为输入值。

### 3.4.2 算法讲解

#### 训练集梯度下降估计回归系数

我们可以使用随机梯度下降来估计训练数据的权值。随机梯度下降需要两个参数:

- **Learning Rate**:用于限制每个重量的数量，每次更新时修正。
- **Epochs**：更新权重时，遍历训练集的次数。

以上参数及训练数据将作为函数的参数。我们将在函数中执行3个循环：

- 每个Epoch循环一次
- 对于每一个Epoch，循环遍历训练集中的每一行
- 对于每一个Epoch中的每一行，循环遍历每个权重并更新它

```
如上，对于每一个Epoch，我们都更新了训练数据中每一行的每个权值。这个权值时根据模型产生的误差进行更新的。误差即期望输出值与由候选权值得到的预测值之间的差。
```


每个输入属性都有一个权重，并且以相同的方式更新这些权重。例如:

$$
w(t + 1) = w(t) + learning\space rate × (expected(t))- predicted(t)) × x(t)\tag{1.4}
$$

除了没有输入外，偏差也以类似的方式进行更新，因为它与特定的输入值没有关联:

$$
bias(t + 1) = bias(t) + learning\space rate × (expected(t))- predicted(t))\tag{1.5}
$$

现在我们可以把这些放在一起。下面是一个名为train weights()的函数，它使用的是随机梯度下降法来计算训练数据集的权值。

```c
void train_weights(double **data, int col,double *weights, double l_rate, int n_epoch, int train_size) {
    int i;
    for (i = 0; i < n_epoch; i++) {
        int j = 0;//遍历每一行
        for (j = 0; j < train_size; j++) {
            double yhat = predict(col,data[j], weights);
            double err = data[j][col - 1] - yhat;
            weights[0] += l_rate * err;
            int k;
            for (k = 0; k < col - 1; k++) {
                weights[k + 1] += l_rate * err * data[j][k];
            }
        }
    }
    for (i = 0; i < col; i++) {
        printf("weights[%d]=%f\n",i, weights[i]);
    }
}
```

我们利用如下数据：

```c
int main()
{
    double data[5][5];
    double* ptr[5];
    double weights[5]={1,2,3,4,5};
    for(int i=0;i<5;i++)
    {
        for(int j=0;j<5;j++)
            data[i][j] = i+j;
        ptr[i] = data[i];
    }
    train_weights(ptr, 5,weights, 0.1,100, 4);
}
```

得到如下结果:

```c
weights[0] = 181.000
weights[1] = 322.000
weights[2] = 503.000
weights[3] = 684.000
weights[4] = 865.000
```

#### 由回归系数计算预测值

开发一个可以进行预测的函数。这在随机梯度下降中评估候选权值时都需要用到，在模型完成后，我们希望对测试数据或新数据进行预测。下面是一个名为predict()的函数，它预测给定一组权重的行的输出值。第一个权重总是偏差，因为它是独立的，不对应任何特定的输入值。

```c
double predict(int col,double *array, double *weights) {
    //预测某一行的值
    double activation = weights[0];
    int i;
    for (i = 0; i < col - 1; i++)
        activation += weights[i + 1] * array[i];
    double output = 0.0;
    if (activation >= 0.0)
        output = 1.0;
    else
        output = 0.0;
    return output;
}
```

我们利用如下数据：

```c
int main()
{
    double data[5]={0,1,2,3,4};
    double weights[5]={1,2,3,4,5};
    predict(5,data,weights);
}
```

得到如下结果：

```c
1.0000
```

### 3.4.3 算法代码

我们现在知道了如何实现**感知器算法**，那么我们把它应用到[声纳数据集 sonar.csv](https://aistudio.baidu.com/aistudio/datasetdetail/105756/0)

我们给出链接：https://aistudio.baidu.com/aistudio/datasetdetail/105756/0

#### C语言细节讲解

本节假设您已下载数据集 `sonar.csv`，并且它在当前工作目录中可用。下面我们给出一个完整实例，使用C语言详细讲解每一处细节。我们给出每一个.c文件的所有代码：

##### 1) read_csv.c

该文件代码与前面代码一致，不再重复给出。

##### 2) k_fold.c

该文件代码与前面代码一致，不再重复给出。

##### 3) score.c

该文件代码与前面代码一致，不再重复给出。

##### 4) normalize.c

该文件代码与前面代码一致，不再重复给出。

##### 5) test_prediction.c

```c
#include <stdlib.h>
#include <stdio.h>

extern void train_weights(double **data, int col, double *weights, double l_rate, int n_epoch, int train_size);
extern double predict(int col, double *array, double *weights);

double *get_test_prediction(double **train, double **test, int row, int col, double l_rate, int n_epoch, int n_folds)
{
	double *weights = (double *)malloc(col * sizeof(double));
	int i;
	for (i = 0; i < col; i++)
	{
		weights[i] = 0.0;
	}
	int fold_size = (int)row / n_folds;
	int train_size = fold_size * (n_folds - 1);
	train_weights(train, col, weights, l_rate, n_epoch, train_size);
	double *predictions = (double *)malloc(fold_size * sizeof(double));
	for (i = 0; i < fold_size; i++)
	{
		predictions[i] = predict(col, test[i], weights);
	}
	return predictions;
}
```

##### 6) evaluate.c

```c
#include <stdlib.h>
#include <stdio.h>

extern double *get_test_prediction(double **train, double **test, int row, int col, double l_rate, int n_epoch, int n_folds);
extern double accuracy_metric(double *actual, double *predicted, int fold_size);
extern double ***cross_validation_split(double **dataset, int row, int col, int n_folds, int fold_size);

double *evaluate_algorithm(double **dataset, int row, int col, int n_folds, int n_epoch, double l_rate)
{
	int fold_size = (int)row / n_folds;
	double ***split = cross_validation_split(dataset, row, n_folds, fold_size, col);
	int i, j, k, l;
	int test_size = fold_size;
	int train_size = fold_size * (n_folds - 1);
	double *score = (double *)malloc(n_folds * sizeof(double));

	for (i = 0; i < n_folds; i++)
	{
		double ***split_copy = (double ***)malloc(n_folds * sizeof(double **));
		for (j = 0; j < n_folds; j++)
		{
			split_copy[j] = (double **)malloc(fold_size * sizeof(double *));
			for (k = 0; k < fold_size; k++)
			{
				split_copy[j][k] = (double *)malloc(col * sizeof(double));
			}
		}
		for (j = 0; j < n_folds; j++)
		{
			for (k = 0; k < fold_size; k++)
			{
				for (l = 0; l < col; l++)
				{
					split_copy[j][k][l] = split[j][k][l];
				}
			}
		}
		double **test_set = (double **)malloc(test_size * sizeof(double *));
		for (j = 0; j < test_size; j++)
		{
			test_set[j] = (double *)malloc(col * sizeof(double));
			for (k = 0; k < col; k++)
			{
				test_set[j][k] = split_copy[i][j][k];
			}
		}
		for (j = i; j < n_folds - 1; j++)
		{
			split_copy[j] = split_copy[j + 1];
		}
		double **train_set = (double **)malloc(train_size * sizeof(double *));
		for (k = 0; k < n_folds - 1; k++)
		{
			for (l = 0; l < fold_size; l++)
			{
				train_set[k * fold_size + l] = (double *)malloc(col * sizeof(double));
				train_set[k * fold_size + l] = split_copy[k][l];
			}
		}
		double *predicted = (double *)malloc(test_size * sizeof(double));
		predicted = get_test_prediction(train_set, test_set, row, col, l_rate, n_epoch, n_folds);
		double *actual = (double *)malloc(test_size * sizeof(double));
		for (l = 0; l < test_size; l++)
		{
			actual[l] = test_set[l][col - 1];
		}
		double accuracy = accuracy_metric(actual, predicted, test_size);
		score[i] = accuracy;
		printf("score[%d] = %f%%\n", i, score[i]);
		free(split_copy);
	}
	double total = 0;
	for (l = 0; l < n_folds; l++)
	{
		total += score[l];
	}
	printf("mean_accuracy = %f%%\n", total / n_folds);
	return score;
}
```

##### 7) main.c

```c
#include <stdlib.h>
#include <stdio.h>

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern double *evaluate_algorithm(double **dataset, int row, int col, int n_folds, int n_epoch, double l_rate);
extern void normalize_dataset(double **dataset, int row, int col);

double predict(int col, double *array, double *weights)
{
	double activation = weights[0];
	int i;
	for (i = 0; i < col - 1; i++)
		activation += weights[i + 1] * array[i];
	double output = 0.0;
	if (activation >= 0.0)
		output = 1.0;
	else
		output = 0.0;
	return output;
}

void train_weights(double **data, int col, double *weights, double l_rate, int n_epoch, int train_size)
{
	int i;
	for (i = 0; i < n_epoch; i++)
	{
		int j = 0;
		for (j = 0; j < train_size; j++)
		{
			double yhat = predict(col, data[j], weights);
			double err = data[j][col - 1] - yhat;
			weights[0] += l_rate * err;
			int k;
			for (k = 0; k < col - 1; k++)
			{
				weights[k + 1] += l_rate * err * data[j][k];
			}
		}
	}
}

int main()
{
    double **dataset;
	int row, col;
	char filename[] = "sonar.csv";
	char line[1024];
	row = get_row(filename);
	col = get_col(filename);
	dataset = (double **)malloc(row * sizeof(double *));
	for (int i = 0; i < row; ++i)
	{
		dataset[i] = (double *)malloc(col * sizeof(double));
	}
	get_two_dimension(line, dataset, filename);
	normalize_dataset(dataset, row, col);
	int n_folds = 3;
	double l_rate = 0.01f;
	int n_epoch = 500;
	evaluate_algorithm(dataset, row, col, n_folds, n_epoch, l_rate);
	return 0;
}
```

##### 8) compile.sh

```bash
gcc main.c read_csv.c normalize.c k_fold.c evaluate.c score.c test_prediction.c -o run -lm && ./run
```

**编译&运行：**

```bash
bash compile.sh
```

最终输出结果如下：

```c
score[0] = 82.608696%
score[1] = 79.710145%
score[2] = 73.913043%
mean_accuracy = 78.743961%
```

#### Python语言实战

本节同样假设您已经下载数据集，我们使用著名机器学习开源库sklearn高效实现**感知机算法**，以便您在实战中使用该算法：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':
    dataset = np.array(pd.read_csv("sonar.csv", sep=',', header=None))
    k_Cross = KFold(n_splits=3, random_state=8, shuffle=True)
    index = 0
    score = np.array([])
    Scaler = MinMaxScaler()
    data,label = dataset[:,:-1],dataset[:,-1]
    data = Scaler.fit_transform(data)
    for train_index, test_index in k_Cross.split(dataset):
        train_data, train_label = data[train_index, :], label[train_index]
        test_data, test_label = data[test_index, :], label[test_index]
        model = Perceptron(eta0=0.01,max_iter=500)
        model.fit(train_data, train_label)
        pred = model.predict(test_data)
        acc = accuracy_score(test_label, pred)
        score = np.append(score,acc)
        print('score[{}] = {}%'.format(index,acc))
        index+=1
    print('mean_accuracy = {}%'.format(np.mean(score)))
```

输出结果如下：

```python
score[0] = 0.7571428571428571%
score[1] = 0.8405797101449275%
score[2] = 0.6956521739130435%
mean_accuracy = 0.7644582470669427%
```

