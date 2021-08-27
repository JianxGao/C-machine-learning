## 3.3 Logistic Regression

> 逻辑回归是一种针对两类问题的归一化线性分类算法。它易于实现，易于理解，并且在各种各样的问题上都能得到很好的结果。在本教程中，您将了解如何借助C语言，使用随机梯度下降从零开始实现逻辑回归

### 3.3.1 算法介绍

#### 逻辑回归

逻辑回归是根据该方法的核心函数——Logistic函数而命名的。逻辑回归可以使用一个方程来表示，与线性回归非常相似。输入值(X)加以权重值或系数值，然后通过它们的线性组合来预测输出值(y)。逻辑与线性回归的一个关键区别是，建模的输出值是二进制值(0或1)，而不是数值。

$$
yhat = \frac{e^{b0+b1×x1}}{1+e^{b0+b2×x1}}\tag{1.1}
$$

可以简化为：

$$
yhat = \frac{1.0}{1.0+e^{-(b0+b2×x1)}}\tag{1.2}
$$

其中e为自然对数(欧拉数)的底数，yhat为预测输出，b0为偏置或截距项，b1为单个输入值(x1)的系数。预测值ythat是一个0到1之间的实值，需要四舍五入到一个整数值并映射到一个预测的类值。

输入数据中的每一列都有一个相关的b系数(一个常实值)，它必须从训练数据中获得。您存储在内存或文件中的模型实际是方程中的系数(beta值或b)。Logistic回归算法的系数由训练数据估计而来。

#### 随机梯度下降

Logistic回归使用梯度下降更新系数。梯度下降在**8.1.2**节中进行了介绍和描述。每次梯度下降迭代时，机器学习语言中的系数(b)根据以下公式更新:

$$
b = b+learning\space rate × (y-yhat)×yhat×(1-yhat)×x\tag{1.3}
$$

其中，b是将被优化的系数或权重，learning rate一个学习速率,它需要您的配置(例如0.01),(y - yhat)是模型在被分配有权重的训练集上的误差,yhat是由预测系数得到的预测值，x是输入值。

### 3.3.2 算法讲解

#### 训练集梯度下降估计回归系数

我们可以使用随机梯度下降估计训练数据的系数值。随机梯度下降需要两个参数:

- **Learning Rate**:用于限制每次更新时每个系数的修正量。
- **Epochs**:更新系数时，训练数据运行的次数。

以上参数及训练数据将作为函数的参数。我们将在函数中执行3个循环：

- 每个Epoch循环一次
- 对于每一个Epoch，循环遍历训练集中的每一行
- 对于每一个Epoch中的每一行，循环遍历每个权重并更新它

如你所见，每个Ephoch，我们根据模型产生的误差更新了训练数据中每一行的系数。误差计算为期望输出值与用候选系数得到的预测值之间的差。每个输入属性有一个权重系数，并且以一致的方式更新，例如:

$$
b1(t+1)=b1(t)+learning\space rate ×(y(t)-yhat(t)×yhat(t)×(1-yhat(t))×x1(t)\tag{1.4}
$$

列表开始处的特殊系数，也称为截距，以类似的方式更新，只是没有输入，因为它与特定的输入值没有关联:

$$
b0(t+1)=b0(t)+learning\space rate ×(y(t)-yhat(t)×yhat(t)×(1-yhat(t))\tag{1.4}
$$

现在我们可以把这些放在一起。下面是一个名为coefficients_sgd()的函数，它使用随机梯度下降法计算训练数据集的系数值。

```c
void coefficients_sgd(double ** dataset, int col, double *coef, double l_rate, int n_epoch, int train_size) {
    int i;
    for (i = 0; i < n_epoch; i++) {
        int j = 0;//遍历每一行
        for (j = 0; j < train_size; j++) {
            double yhat = predict(col,dataset[j], coef);
            double err = dataset[j][col - 1] - yhat;
            coef[0] += l_rate * err * yhat * (1 - yhat);
            int k;
            for (k = 0; k < col - 1; k++) {
                coef[k + 1] += l_rate * err * yhat * (1 - yhat) * dataset[j][k];
            }
        }
    }
    for (i = 0; i < col; i++) {
        printf("coef[%d]=%f\n",i, coef[i]);
    }
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

在随机梯度下降中评估候选系数值时，以及在模型完成并且我们希望开始对测试数据或新数据进行预测后，都需要一个可以预测的函数。下面是一个名为predict()的函数，给定一组系数后，它可以输出一行预测值。第一个系数是截距，也称为偏差或b0，因为它是独立的，不对应特定的输入值。

```c
double predict(int col, double array[], double coefficients[]) {//预测某一行的值
    double yhat = coefficients[0];
    int i;
    for (i = 0; i < col - 1; i++)
        yhat += coefficients[i + 1] * array[i];
    printf("%f",1 / (1 + exp(-yhat)));
    return 1 / (1 + exp(-yhat));
}
```

我们使用如下数据：

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

### 3.3.3 算法代码

我们现在知道了如何实现一个**逻辑回归模型**，那么我们把它应用到[糖尿病数据集 Pima.csv](https://aistudio.baidu.com/aistudio/datasetdetail/105756/0)

我们给出链接：https://aistudio.baidu.com/aistudio/datasetdetail/105756/0

#### C语言细节讲解

本节假设您已下载数据集 `Pima.csv`，并且它在当前工作目录中可用。下面我们给出一个完整实例，使用C语言详细讲解每一处细节。我们给出每一个.c文件的所有代码：

##### 1) read_csv.c

该文件代码与前面代码一致，不再重复给出。

##### 2) k_fold.c

该文件代码与前面代码一致，不再重复给出。

##### 3) score.c

```c
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double accuracy_metric(double *actual, double *predicted, int fold_size)
{
	int correct = 0;
	int i;
	for (i = 0; i < fold_size; i++)
	{
		if (actual[i] == predicted[i])
			correct += 1;
	}
	return (correct / (double)fold_size) * 100.0;
}
```

##### 4) normalize.c

该文件代码与前面代码一致，不再重复给出。

##### 5) test_prediction.c

```c
#include <stdlib.h>
#include <stdio.h>

extern void coefficients_sgd(double **dataset, int col, double *coef, double l_rate, int n_epoch, int train_size);
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
		predictions[i] = (float)(int)(predictions[i] + 0.5);
	}
	return predictions;
}
```

##### 6) evaluate.c

```c
#include <stdlib.h>
#include <stdio.h>

extern double *get_test_prediction(double **dataset, int row, int col, double **train, double **test, double l_rate, int n_epoch, int n_folds);
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
		// 因为要遍历删除，所以拷贝一份split
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
		// 删除取出来的fold
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
		double accuracy = accuracy_metric(actual, predicted, test_size);
		score[i] = accuracy;
		printf("score[%d]=%f%%\n", i, score[i]);
		free(split_copy);
	}
	double total = 0;
	for (l = 0; l < n_folds; l++)
	{
		total += score[l];
	}
	printf("mean_accuracy=%f%%\n", total / n_folds);
	return score;
}
```

##### 7) main.c

```c
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double **dataset;
int row, col;

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern double *evaluate_algorithm(double **dataset, int row, int col, int n_folds, int n_epoch, double l_rate);
extern void normalize_dataset(double **dataset, int row, int col);

void coefficients_sgd(double **dataset, int col, double *coef, double l_rate, int n_epoch, int train_size)
{
	int i;
	for (i = 0; i < n_epoch; i++)
	{
		int j = 0;
		for (j = 0; j < train_size; j++)
		{
			double yhat = predict(col, dataset[j], coef);
			double err = dataset[j][col - 1] - yhat;
			coef[0] += l_rate * err * yhat * (1 - yhat);
			int k;
			for (k = 0; k < col - 1; k++)
			{
				coef[k + 1] += l_rate * err * yhat * (1 - yhat) * dataset[j][k];
			}
		}
	}
}

double predict(int col, double array[], double coefficients[])
{
	double yhat = coefficients[0];
	int i;
	for (i = 0; i < col - 1; i++)
		yhat += coefficients[i + 1] * array[i];
	return 1 / (1 + exp(-yhat));
}

int main()
{
	char filename[] = "Pima.csv";
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
	int n_folds = 5;
	double l_rate = 0.1f;
	int n_epoch = 100;
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
score[0] = 78.431373%
score[1] = 79.738562%
score[2] = 72.549020%
score[3] = 75.163399%
score[4] = 77.124183%
mean_accuracy = 76.601307%
```

#### Python语言实战

本节同样假设您已经下载数据集，我们使用著名机器学习开源库sklearn高效实现**逻辑回归算法**，以便您在实战中使用该算法：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':
    dataset = np.array(pd.read_csv("Pima.csv", sep=',', header=None))
    k_Cross = KFold(n_splits=5, random_state=0, shuffle=True)
    index = 0
    score = np.array([])
    Scaler = MinMaxScaler()
    data,label = dataset[:,:-1],dataset[:,-1]
    data = Scaler.fit_transform(data)
    for train_index, test_index in k_Cross.split(dataset):
        train_data, train_label = data[train_index, :], label[train_index]
        test_data, test_label = data[test_index, :], label[test_index]
        model = LogisticRegression()
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
score[0] = 0.8181818181818182%
score[1] = 0.7532467532467533%
score[2] = 0.7467532467532467%
score[3] = 0.7843137254901961%
score[4] = 0.7581699346405228%
mean_accuracy = 0.7721330956625074%
```

#### 