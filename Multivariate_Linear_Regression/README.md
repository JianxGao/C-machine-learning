## 3.2 Multivariate Linear Regression

> 许多机器学习算法的核心是优化。优化算法通常被用来从给定的数据集中寻找一个良好的模型参数。机器学习中最常用的优化算法是随机梯度下降法。在本教程中，您将了解如何使用C语言从头开始实现随机梯度下降，并以此来优化线性回归算法。

### 3.2.1 算法介绍

#### 多元线性回归

线性回归是一种预测真实值的算法。这些需要预测真实值的问题叫做回归问题。线性回归是一种使用直线来建立输入值和输出值之间关系模型的算法。在二维以上的空间中，这条直线被称为一个平面或超平面。

预测就是通过输入值的组合来预测输出值。每个输入属性(x)都使用一个系数(b)对其进行加权，学习算法的目标遍是寻找这组能产生良好预测(y)的系数。

$$
y = b_0+b_1×x_1+b_2×x_1+...\tag{1.1}
$$

这组系数可以用随机梯度下降法求得。

#### 随机梯度下降

梯度下降是沿着函数的斜率或梯度最小化函数的过程。在机器学习中，我们可以使用一种算法来评估和更新每次迭代的效率，以最小化我们的模型在训练数据上的误差，这种算法被称为随机梯度下降法。

这种优化算法的工作原理是每次向模型提供一个训练样本，模型对这个训练样本进行预测，计算误差并更新模型以减少下一次预测的误差。此迭代过程将重复进行固定次数。

当我们得到某个模型的一个系数后，以下方程可以用来由已知系数计算新系数，从而使模型在训练数据上的误差最小。每次迭代，该方程都会使模型中的系数(b)更新。

$$
b = b-learning\space rate × error × x\tag{1.2}
$$

其中，b是被优化的系数或权值，学习率是你需要设定的参数(例如0.01)，error是模型由于权值问题导致在训练集上的预测误差，x是输入变量。

### 3.2.2 算法讲解

#### 训练集梯度下降估计回归系数

我们可以使用随机梯度下降法来估计训练数据的系数。随机梯度下降法需要两个参数：

- **Learning Rate**:用于限制每次更新时每个系数的修正量。
- **Epochs**:更新系数时遍历训练数据的次数。

以上参数及训练数据将作为函数的输入参数。我们需要在函数中执行以下3个循环:

- 循环每个Epoch。
- 对于每个Epoch，循环训练数据中的每一行。
- 对于每个Epoch中的每一行，循环并更新每个系数。

我们可以看到，对于每个Epochs，我们更新了训练数据中每一行的系数。这种更新是根据模型的误差产生的。其中，误差是用候选系数的预测值与预期输出值的差来计算的。

$$
error = prediction-expected\tag{1.3}
$$

每个输入属性都有一个权重系数，并且以相同的方式更新，例如:

$$
b_1(t+1) = b_1(t) - learning\space rate × error(t) × x_1(t)\tag{1.4}
$$

迭代最初的系数，也称为截距或偏置，同样以类似的方式更新。最终结果与系数的初始输入值无关。

$$
b_0(t+1) = b_0(t) - learning\space rate × error(t)\tag{1.5}
$$

下面是一个名为coefficients sgd()的函数，它使用随机梯度下降法计算训练数据集的系数。

- 功能——估计回归系数
- 以训练集数组、数组列数、系数存储数组、学习率、epoch、训练集长度作为输入参数。
- 最终输出系数存储数组。

```c
double* coefficients_sgd(double** dataset,int col,double coef[], double l_rate, int n_epoch, int train_size) {
    int i;
    for (i = 0; i < n_epoch; i++) {
        int j = 0;//遍历每一行
        for (j = 0; j < train_size; j++) {
            double yhat = predict(col,dataset[j], coef);
            double err = yhat - dataset[j][col - 1];
            coef[0] -= l_rate * err;
            int k;
            for (k = 0; k < col - 1; k++) {
                coef[k + 1] -= l_rate * err*dataset[j][k];
            }
        }
    }
    for (i = 0; i < col; i++) {
        printf("coef[i]=%f\n", coef[i]);
    }
    return coef;
}
```

我们利用如下数据集：

```c
int main()
{
    double data[5][5];
    double* ptr[5];
    double weight[5]={1,2,3,4,5};
    for(int i=0;i<5;i++)
    {
        for(int j=0;j<5;j++)
            data[i][j] = i+j;
        ptr[i] = data[i];
    }
    coefficients_sgd(ptr, 5,weight, 0.1,100, 4);
}
```

得到结果如下:

```c
weights[0] = 181.000
weights[1] = 322.000
weights[2] = 503.000
weights[3] = 684.000
weights[4] = 865.000
```

#### 由回归系数计算预测值

在随机梯度下降中评估候选系数值时，在模型完成并且我们想要开始对测试数据或新数据进行预测时，我们都需要一个预测函数。

例如，有一个输入值(x)和两个系数值(b0和b1)。这个问题建模的预测方程为:

$$
y = b_0+b_1×x\tag{1.6}
$$

以下是一个名为predict()的预测函数，给定系数后，它可以预测一组输入值(x)对应的输出值(y)。

- 功能——预测输出值(y)

- 以输入值的属性个数、输入值、系数数组为输入参数

- 最终输出预测值

- ```c
  double predict(int col,double array[], double coefficients[]) {//预测某一行的值
      double yhat = coefficients[0];
      int i;
      for (i = 0; i < col - 1; i++)
          yhat += coefficients[i + 1] * array[i];
      return yhat;
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

```
1.0000
```

### 3.2.3 算法代码

我们现在知道了如何实现**多元线性回归算法**，那么我们把它应用到[葡萄酒质量数据集 winequize-white.csv](https://aistudio.baidu.com/aistudio/datasetdetail/105756/0)

我们给出链接：https://aistudio.baidu.com/aistudio/datasetdetail/105756/0

#### C语言细节讲解

本节假设您已下载数据集 `winequize-white.csv`，并且它在当前工作目录中可用。下面我们给出一个完整实例，使用C语言详细讲解每一处细节。我们给出每一个.c文件的所有代码：


##### 1) read_csv.c

该文件代码与前面代码一致，不再重复给出。

##### 2) k_fold.c

该文件代码与前面代码一致，不再重复给出。

##### 3) rmse.c

该文件代码与前面代码一致，不再重复给出。

##### 4) normalize.c

```c
void normalize_dataset(double **dataset,int row, int col) 
{
    // 先 对列循环
    double maximum, minimum;
    for (int i = 0; i < col; i++) 
    {
        // 第一行为标题，值为0，不能参与计算最大最小值
        maximum = dataset[0][i];
        minimum = dataset[0][i];
        //再 对行循环
        for (int j = 0; j < row; j++) 
        {
            maximum = (dataset[j][i]>maximum)?dataset[j][i]:maximum;
            minimum = (dataset[j][i]<minimum)?dataset[j][i]:minimum;
        }
        // 归一化处理
        for (int j = 0; j < row; j++)
        {
            dataset[j][i] = (dataset[j][i] - minimum) / (maximum - minimum);
        }
    }
}
```

##### 5) test_prediction.c

```c
#include <stdlib.h>
#include <stdio.h>

extern double *coefficients_sgd(double **dataset, int col, double coef[], double l_rate, int n_epoch, int train_size);
extern double predict(int col, double array[], double coefficients[]);

double *get_test_prediction(double **dataset, int row, int col, double **train, double **test, double l_rate, int n_epoch, int n_folds)
{
	double *coef = (double *)malloc(col * sizeof(double));
	int i;
	for (i = 0; i < col; i++)
	{
		coef[i] = 0.0;
	}
	int fold_size = (int)row / n_folds;
	int train_size = fold_size * (n_folds - 1);
	coefficients_sgd(train, col, coef, l_rate, n_epoch, train_size);
	double *predictions = (double *)malloc(fold_size * sizeof(double)); 
	for (i = 0; i < fold_size; i++)
	{
		predictions[i] = predict(col, test[i], coef);
	}
	return predictions;
}
```

##### 6) evaluate.c

```c
#include <stdlib.h>
#include <stdio.h>

extern double *get_test_prediction(double **dataset, int row, int col, double **train, double **test, double l_rate, int n_epoch, int n_folds);
extern double rmse_metric(double *actual, double *predicted, int fold_size);
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
		//因为要遍历删除，所以拷贝一份split
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
		} //删除取出来的fold

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
		predicted = get_test_prediction(dataset, row, col, train_set, test_set, l_rate, n_epoch, n_folds);
		double *actual = (double *)malloc(test_size * sizeof(double));
		for (l = 0; l < test_size; l++)
		{
			actual[l] = test_set[l][col - 1];
		}
		double rmse = rmse_metric(actual, predicted, test_size);
		score[i] = rmse;
		printf("score[%d] = %f\n", i, score[i]);
		free(split_copy);
	}
	double total = 0;
	for (l = 0; l < n_folds; l++)
	{
		total += score[l];
	}
	printf("mean_rmse = %f\n", total / n_folds);
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

double *coefficients_sgd(double **dataset, int col, double coef[], double l_rate, int n_epoch, int train_size)
{
	int i;
	for (i = 0; i < n_epoch; i++)
	{
		int j = 0;
		for (j = 0; j < train_size; j++)
		{
			double yhat = predict(col, dataset[j], coef);
			double err = yhat - dataset[j][col - 1];
			coef[0] -= l_rate * err;
			int k;
			for (k = 0; k < col - 1; k++)
			{
				coef[k + 1] -= l_rate * err * dataset[j][k];
			}
		}
	}
	return coef;
}

double predict(int col, double array[], double coefficients[])
{
	double yhat = coefficients[0];
	int i;
	for (i = 0; i < col - 1; i++)
		yhat += coefficients[i + 1] * array[i];
	return yhat;
}

int main()
{
	char filename[] = "winequality-white.csv";
	char line[1024];
	double **dataset;
	int row, col;
	row = get_row(filename);
	col = get_col(filename);
	dataset = (double **)malloc(row * sizeof(double *));
	for (int i = 0; i < row; ++i)
	{
		dataset[i] = (double *)malloc(col * sizeof(double));
	}
	get_two_dimension(line, dataset, filename);
	normalize_dataset(dataset, row, col);

	int n_folds = 10;
	double l_rate = 0.001f;
	int n_epoch = 50;
	evaluate_algorithm(dataset, row, col, n_folds, n_epoch, l_rate);
	return 0;
}
```

##### 8) compile.sh

```bash
gcc main.c read_csv.c normalize.c k_fold.c evaluate.c rmse.c test_prediction.c -o run -lm && ./run
```

**编译&运行：**

```bash
bash compile.sh
```

最终输出结果如下：

```c
score[0] = 0.221540
score[1] = 0.209277
score[2] = 0.221540
score[3] = 0.219608
score[4] = 0.219479
score[5] = 0.216744
score[6] = 0.205718
score[7] = 0.202798
score[8] = 0.214637
score[9] = 0.207231
mean_rmse = 0.213857
```

#### Python语言实战

本节同样假设您已经下载数据集，我们使用著名机器学习开源库sklearn高效实现**多元线性回归算法**，以便您在实战中使用该算法：

```python
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
```

输出结果如下，读者可以尝试分析一下为何结果会存在差异？

```python
score[0] = 0.8419014234018158
score[1] = 0.8408919161041173
score[2] = 0.7311825499558641
score[3] = 0.8147707681816518
score[4] = 0.7276314042865725
score[5] = 0.7403970929333936
score[6] = 0.7865008610855795
score[7] = 0.822294388008359
score[8] = 0.7899616477361368
score[9] = 0.7447500726966548
mean_rmse = 0.7840282124390145
```
