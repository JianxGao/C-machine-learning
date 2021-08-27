## 3.13 Stacked Generalization

> David H. Wolpert在1992年发表了SG算法论文，正式提出SG算法。该算法属于集成学习，集成一系列子模型，提高了机器学习的准确率。

### 3.13.1 算法介绍

SG算法，即堆栈泛化。堆栈泛化算法使用一系列的子模型

$$
(m_1,m_2,\dots,m_n)
$$

每一个子模型分别对数据集进行分类/回归预测，得到结果：

$$
(r_1,r_2,\dots,m_n)
$$

将每一个样本的 $n$ 个预测值组成新的 $stack\_row$，从而生成一个新的数据集，再使用一个新的模型 $(Aggregator Model)$ 对新的数据集进行分类/回归预测，完成算法。

### 3.13.2 算法讲解

本节以子模型分别为 $KNN、Perceptron$，集成模型为 $Logistic\ Regression$

#### KNN子模型

该部分代码主体与前面章节介绍的代码一致。

【注意】

- 函数名需要相应的改变
- 此时KNN用于分类任务，故预测值为近邻类别的众数

```c
#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<math.h>

void QuickSort(double **arr, int L, int R) {
    int i = L;
    int j = R;
    //支点
    int kk = (L + R) / 2;
    double pivot = arr[kk][0];
    //左右两端进行扫描，只要两端还没有交替，就一直扫描
    while (i <= j) {
        //寻找直到比支点大的数
        while (pivot > arr[i][0])
        {
            i++;
        }
        //寻找直到比支点小的数
        while (pivot < arr[j][0])
        {
            j--;
        }
        //此时已经分别找到了比支点小的数(右边)、比支点大的数(左边)，它们进行交换
        if (i <= j) {
            double *temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            //double temp2 = arr[i][1];
            //arr[i][1] = arr[j][1];
            //arr[j][1] = temp2;
            i++;
            j--;
        }
    }//上面一个while保证了第一趟排序支点的左边比支点小，支点的右边比支点大了。
    //“左边”再做排序，直到左边剩下一个数(递归出口)
    if (L < j)
    {
        QuickSort(arr, L, j);
    }
    //“右边”再做排序，直到右边剩下一个数(递归出口)
    if (i < R)
    {
        QuickSort(arr, i, R);
    }
}
double euclidean_distance(double *row1, double *row2, int col) {
    double distance = 0;
    for (int i = 0; i < col - 1; i++) {
        distance += pow((row1[i] - row2[i]), 2);
    }
    return distance;
}
double* get_neighbors(double **train_data, int train_row, int col, double *test_row, int num_neighbors) {
    double *neighbors = (double *)malloc(num_neighbors * sizeof(double));
    double **distances = (double **)malloc(train_row * sizeof(double *));
    for (int i = 0; i < train_row; i++) {
        distances[i] = (double *)malloc(2 * sizeof(double));
        distances[i][0] = euclidean_distance(train_data[i], test_row, col);
        distances[i][1] = train_data[i][col - 1];
    }
    QuickSort(distances, 0, train_row - 1);
    for (int i = 0; i < num_neighbors; i++) {
        neighbors[i] = distances[i][1];
    }
    return neighbors;
}
double knn_single_predict(double **train_data, int train_row, int col, double *test_row, int num_neighbors) {
    double* neighbors = get_neighbors(train_data, train_row, col, test_row, num_neighbors);
    double result = 0;
    for (int i = 0; i < num_neighbors; i++) {
        result += neighbors[i];
    }
    return round(result / num_neighbors);
}


double* knn_predict(double **train, int train_size, double **test, int test_size, int col, int num_neighbors)
{
    double* predictions = (double*)malloc(test_size * sizeof(double));
    for (int i = 0; i < test_size; i++)
    {
        predictions[i] = knn_single_predict(train, train_size, col, test[i], num_neighbors);
    }
    return predictions;
}
```

#### Perceptron子模型

该部分代码主体与前面章节介绍的代码一致。

【注意】

- 函数名需要相应的改变

```c
#include<stdlib.h>
#include<stdio.h>

double perceptron_single_predict(int col, double *array, double *weights) {
    double activation = weights[0];
    for (int i = 0; i < col - 1; i++)
    {
        activation += weights[i + 1] * array[i];
    }
    double output = 0.0;
    if (activation >= 0.0)
    {
        output = 1.0;
    }
    else
    {
        output = 0.0;
    }
    return output;
}

void train_weights(double **data, int col, double *weights, double l_rate, int n_epoch, int train_size) {
    for (int i = 0; i < n_epoch; i++) 
    {
        for (int j = 0; j < train_size; j++) 
        {
            double yhat = perceptron_single_predict(col, data[j], weights);
            double err = data[j][col - 1] - yhat;
            weights[0] += l_rate * err;
            for (int k = 0; k < col - 1; k++) 
            {
                weights[k + 1] += l_rate * err * data[j][k];
            }
        }
    }
}

double* perceptron_predict(double** train, int train_size, double** test, int test_size, int col, double l_rate, int n_epoch) {
    double* weights = (double*)malloc(col * sizeof(double));
    int i;
    for (i = 0; i < col; i++) {
        weights[i] = 0.0;
    }
    train_weights(train, col, weights, l_rate, n_epoch, train_size);//核心算法执行部分
    double* predictions = (double*)malloc(test_size * sizeof(double));//因为test_size和fold_size一样大
    for (i = 0; i < test_size; i++) {//因为test_size和fold_size一样大
        predictions[i] = perceptron_single_predict(col, test[i], weights);
    }
    return predictions;//返回对test的预测数组
}
```

#### 2.3  Logistic Regression集成模型

该部分代码主体与前面章节介绍的代码一致。

【注意】

- 函数名需要相应的改变

```c
#include<stdlib.h>
#include<stdio.h>
#include<math.h>

double stack_predict(int col, double *array, double *coefficients)
{
    double yhat = coefficients[0];
    int i;
    for (i = 0; i < col - 1; i++){
        yhat += coefficients[i + 1] * array[i];
    }
    return 1 / (1 + exp(-yhat));
}

void coefficients_sgd(double ** dataset, int col, double *coef, double l_rate, int n_epoch, int train_size) 
{
    for (int i = 0; i < n_epoch; i++)
    {
        for (int j = 0; j < train_size; j++)
        {
            double yhat = stack_predict(col, dataset[j], coef);
            double err = dataset[j][col - 1] - yhat;
            coef[0] += l_rate * err * yhat * (1 - yhat);

            for (int k = 0; k < col - 1; k++)
            {
                coef[k + 1] += l_rate * err * yhat * (1 - yhat) * dataset[j][k];
            }
        }
    }
}
```

#### 2.4  利用KNN与Perceptron的结果，预测结果

```c
#include<stdlib.h>
#include<stdio.h>
#include<math.h>

extern double* knn_predict(double **train, int train_size, double **test, int test_size, int col, int num_neighbors);
extern double* perceptron_predict(double** train, int train_size, double** test, int test_size, int col, double l_rate, int n_epoch);
extern void coefficients_sgd(double ** dataset, int col, double *coef, double l_rate, int n_epoch, int train_size);
extern double stack_predict(int col, double *array, double *coefficients);

double* get_test_prediction(double **train, int train_size, double **test, int test_size, int col, int num_neighbors, double l_rate, int n_epoch)
{
    double* coef = (double*)malloc(3 * sizeof(double));
    for (int i = 0; i < 3; i++)
    {
        coef[i] = 0.0;
    }
    // 训练
    double* train_knn_predictions = knn_predict(train, train_size, train, train_size, col, num_neighbors);
    double* train_perceptron_predictions = perceptron_predict(train, train_size, train, train_size, col, l_rate, n_epoch);
    double** new_train = (double **)malloc(train_size * sizeof(double *));
    for (int i = 0; i < train_size; i++)
    {
        new_train[i] = (double *)malloc(3 * sizeof(double));
    }
    // 生成新的数据集
    for (int i = 0; i < train_size; i++)
    {
        new_train[i][0] = train_knn_predictions[i];
        new_train[i][1] = train_perceptron_predictions[i];
        new_train[i][2] = train[i][col - 1];
    }

    coefficients_sgd(new_train, 3, coef, l_rate, n_epoch, train_size);

    // 预测
    double* knn_predictions = knn_predict(train, train_size, test, test_size, col, num_neighbors);
    double* perceptron_predictions = perceptron_predict(train, train_size, test, test_size, col, l_rate, n_epoch);

    double** new_test = (double **)malloc(test_size * sizeof(double *));
    for (int i = 0; i < test_size; ++i)
    {
        new_test[i] = (double *)malloc(2 * sizeof(double));
    }
    // 生成新的数据集
    for (int i = 0; i < test_size; i++)
    {
        new_test[i][0] = knn_predictions[i];
        new_test[i][1] = perceptron_predictions[i];
    }
    double* predictions = (double*)malloc(test_size * sizeof(double));
    for (int i = 0; i < test_size; i++)
    {
        predictions[i] = round(stack_predict(3, new_test[i], coef));
    }

    return predictions;//返回对test的预测数组
}
```

### 3.13.3 算法代码


我们现在知道了如何实现**堆栈泛化算法**，那么我们把它应用到[声纳数据集 sonar.csv](https://aistudio.baidu.com/aistudio/datasetdetail/105756/0)

我们给出链接：https://aistudio.baidu.com/aistudio/datasetdetail/105756/0

#### C语言细节讲解

本节假设您已下载数据集 `sonar.csv`，并且它在当前工作目录中可用。下面我们给出一个完整实例，使用C语言详细讲解每一处细节。我们给出每一个.c文件的所有代码：


##### 1) read_csv.c

该文件代码与前面代码一致，不再重复给出。

##### 2) normalize.c

该文件代码与前面代码一致，不再重复给出。

##### 3) k_fold.c

该文件代码与前面代码一致，不再重复给出。

##### 4) score.c

该文件代码与前面代码一致，不再重复给出。

##### 5) knn_model.c

```c
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

void QuickSort(double **arr, int L, int R)
{
	int i = L;
	int j = R;
	//支点
	int kk = (L + R) / 2;
	double pivot = arr[kk][0];
	//左右两端进行扫描，只要两端还没有交替，就一直扫描
	while (i <= j)
	{
		//寻找直到比支点大的数
		while (pivot > arr[i][0])
		{
			i++;
		}
		//寻找直到比支点小的数
		while (pivot < arr[j][0])
		{
			j--;
		}
		//此时已经分别找到了比支点小的数(右边)、比支点大的数(左边)，它们进行交换
		if (i <= j)
		{
			double *temp = arr[i];
			arr[i] = arr[j];
			arr[j] = temp;
			//double temp2 = arr[i][1];
			//arr[i][1] = arr[j][1];
			//arr[j][1] = temp2;
			i++;
			j--;
		}
	} //上面一个while保证了第一趟排序支点的左边比支点小，支点的右边比支点大了。
	//“左边”再做排序，直到左边剩下一个数(递归出口)
	if (L < j)
	{
		QuickSort(arr, L, j);
	}
	//“右边”再做排序，直到右边剩下一个数(递归出口)
	if (i < R)
	{
		QuickSort(arr, i, R);
	}
}
double euclidean_distance(double *row1, double *row2, int col)
{
	double distance = 0;
	for (int i = 0; i < col - 1; i++)
	{
		distance += pow((row1[i] - row2[i]), 2);
	}
	return distance;
}
double *get_neighbors(double **train_data, int train_row, int col, double *test_row, int num_neighbors)
{
	double *neighbors = (double *)malloc(num_neighbors * sizeof(double));
	double **distances = (double **)malloc(train_row * sizeof(double *));
	for (int i = 0; i < train_row; i++)
	{
		distances[i] = (double *)malloc(2 * sizeof(double));
		distances[i][0] = euclidean_distance(train_data[i], test_row, col);
		distances[i][1] = train_data[i][col - 1];
	}
	QuickSort(distances, 0, train_row - 1);
	for (int i = 0; i < num_neighbors; i++)
	{
		neighbors[i] = distances[i][1];
	}
	return neighbors;
}
double knn_single_predict(double **train_data, int train_row, int col, double *test_row, int num_neighbors)
{
	double *neighbors = get_neighbors(train_data, train_row, col, test_row, num_neighbors);
	double result = 0;
	for (int i = 0; i < num_neighbors; i++)
	{
		result += neighbors[i];
	}
	return round(result / num_neighbors);
}

double *knn_predict(double **train, int train_size, double **test, int test_size, int col, int num_neighbors)
{
	double *predictions = (double *)malloc(test_size * sizeof(double));
	for (int i = 0; i < test_size; i++)
	{
		predictions[i] = knn_single_predict(train, train_size, col, test[i], num_neighbors);
	}
	return predictions; //返回对test的预测数组
}
```

##### 6) perceptron_model.c

```c
#include <stdlib.h>
#include <stdio.h>

double perceptron_single_predict(int col, double *array, double *weights)
{
	double activation = weights[0];
	for (int i = 0; i < col - 1; i++)
	{
		activation += weights[i + 1] * array[i];
	}
	double output = 0.0;
	if (activation >= 0.0)
	{
		output = 1.0;
	}
	else
	{
		output = 0.0;
	}
	return output;
}

void train_weights(double **data, int col, double *weights, double l_rate, int n_epoch, int train_size)
{
	for (int i = 0; i < n_epoch; i++)
	{
		for (int j = 0; j < train_size; j++)
		{
			double yhat = perceptron_single_predict(col, data[j], weights);
			double err = data[j][col - 1] - yhat;
			weights[0] += l_rate * err;
			for (int k = 0; k < col - 1; k++)
			{
				weights[k + 1] += l_rate * err * data[j][k];
			}
		}
	}
}

double *perceptron_predict(double **train, int train_size, double **test, int test_size, int col, double l_rate, int n_epoch)
{
	double *weights = (double *)malloc(col * sizeof(double));
	int i;
	for (i = 0; i < col; i++)
	{
		weights[i] = 0.0;
	}
	train_weights(train, col, weights, l_rate, n_epoch, train_size);
	double *predictions = (double *)malloc(test_size * sizeof(double));
	for (i = 0; i < test_size; i++)
	{
		predictions[i] = perceptron_single_predict(col, test[i], weights);
	}
	return predictions;
}
```

##### 7) stacking_model.c

```c
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double stack_predict(int col, double *array, double *coefficients)
{
	double yhat = coefficients[0];
	int i;
	for (i = 0; i < col - 1; i++)
	{
		yhat += coefficients[i + 1] * array[i];
	}
	return 1 / (1 + exp(-yhat));
}

void coefficients_sgd(double **dataset, int col, double *coef, double l_rate, int n_epoch, int train_size)
{
	for (int i = 0; i < n_epoch; i++)
	{
		for (int j = 0; j < train_size; j++)
		{
			double yhat = stack_predict(col, dataset[j], coef);
			double err = dataset[j][col - 1] - yhat;
			coef[0] += l_rate * err * yhat * (1 - yhat);

			for (int k = 0; k < col - 1; k++)
			{
				coef[k + 1] += l_rate * err * yhat * (1 - yhat) * dataset[j][k];
			}
		}
	}
}
```

##### 8) test_prediction.c

```c
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

extern double *knn_predict(double **train, int train_size, double **test, int test_size, int col, int num_neighbors);
extern double *perceptron_predict(double **train, int train_size, double **test, int test_size, int col, double l_rate, int n_epoch);
extern void coefficients_sgd(double **dataset, int col, double *coef, double l_rate, int n_epoch, int train_size);
extern double stack_predict(int col, double *array, double *coefficients);

double *get_test_prediction(double **train, int train_size, double **test, int test_size, int col, int num_neighbors, double l_rate, int n_epoch)
{
	double *coef = (double *)malloc(3 * sizeof(double));
	for (int i = 0; i < 3; i++)
	{
		coef[i] = 0.0;
	}
	// 训练
	double *train_knn_predictions = knn_predict(train, train_size, train, train_size, col, num_neighbors);
	double *train_perceptron_predictions = perceptron_predict(train, train_size, train, train_size, col, l_rate, n_epoch);
	double **new_train = (double **)malloc(train_size * sizeof(double *));
	for (int i = 0; i < train_size; i++)
	{
		new_train[i] = (double *)malloc(3 * sizeof(double));
	}
	// 生成新的数据集
	for (int i = 0; i < train_size; i++)
	{
		new_train[i][0] = train_knn_predictions[i];
		new_train[i][1] = train_perceptron_predictions[i];
		new_train[i][2] = train[i][col - 1];
	}

	coefficients_sgd(new_train, 3, coef, l_rate, n_epoch, train_size);

	// 预测
	double *knn_predictions = knn_predict(train, train_size, test, test_size, col, num_neighbors);
	double *perceptron_predictions = perceptron_predict(train, train_size, test, test_size, col, l_rate, n_epoch);

	double **new_test = (double **)malloc(test_size * sizeof(double *));
	for (int i = 0; i < test_size; ++i)
	{
		new_test[i] = (double *)malloc(2 * sizeof(double));
	}
	// 生成新的数据集
	for (int i = 0; i < test_size; i++)
	{
		new_test[i][0] = knn_predictions[i];
		new_test[i][1] = perceptron_predictions[i];
	}
	double *predictions = (double *)malloc(test_size * sizeof(double)); //因为test_size和fold_size一样大
	for (int i = 0; i < test_size; i++)
	{
		predictions[i] = round(stack_predict(3, new_test[i], coef));
	}
	return predictions; //返回对test的预测数组
}
```

##### 9) evaluate.c

```c
#include <stdlib.h>
#include <stdio.h>

extern double *get_test_prediction(double **train, int train_size, double **test, int test_size, int col, int num_neighbors, double l_rate, int n_epoch);
extern double accuracy_metric(double *actual, double *predicted, int fold_size);
extern double ***cross_validation_split(double **dataset, int row, int col, int n_folds, int fold_size);

void evaluate_algorithm(double **dataset, int row, int col, int n_folds, int num_neighbors, double l_rate, int n_epoch)
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
		predicted = get_test_prediction(train_set, train_size, test_set, test_size, col, num_neighbors, l_rate, n_epoch);
		double *actual = (double *)malloc(test_size * sizeof(double));
		for (l = 0; l < test_size; l++)
		{
			actual[l] = test_set[l][col - 1];
		}
		double acc = accuracy_metric(actual, predicted, test_size);
		score[i] = acc;
		printf("Scores[%d] = %f%%\n", i, score[i]);
		free(split_copy);
	}
	double total = 0;
	for (l = 0; l < n_folds; l++)
	{
		total += score[l];
	}
	printf("mean_accuracy = %f%%\n", total / n_folds);
}
```

##### 10) main.c

```c
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern void normalize_dataset(double **dataset, int row, int col);
extern void evaluate_algorithm(double **dataset, int row, int col, int n_folds, int num_neighbors, double l_rate, int n_epoch);

void main()
{
	char filename[] = "sonar.csv";
	char line[1024];
	int row = get_row(filename);
	int col = get_col(filename);
	double **dataset;
	dataset = (double **)malloc(row * sizeof(double *));
	for (int i = 0; i < row; ++i)
	{
		dataset[i] = (double *)malloc(col * sizeof(double));
	}
	get_two_dimension(line, dataset, filename);
	normalize_dataset(dataset, row, col);
	int k_fold = 3;
	int num_neighbours = 2;
	double l_rate = 0.01;
	int n_epoch = 5000;
	evaluate_algorithm(dataset, row, col, k_fold, num_neighbours, l_rate, n_epoch);
}
```

##### 11) compile.sh

```bash
gcc main.c normalize.c score.c test_prediction.c stacking_model.c k_fold.c knn_model.c evaluate.c read_csv.c perceptron_model.c -o test -lm && ./run
```

**编译&运行：**

```bash
bash compile.sh
```

运算后得到的结果如下：

```c
Scores[0] = 82.608696%
Scores[1] = 79.710145%
Scores[2] = 73.913043%
mean_accuracy = 78.743961%
```


#### Python语言实战

本节同样假设您已经下载数据集，我们使用著名机器学习开源库sklearn高效实现**堆栈泛化算法**，以便您在实战中使用该算法：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from mlxtend.classifier import StackingClassifier


if __name__ == '__main__':
    dataset = np.array(pd.read_csv("sonar.csv", sep=',', header=None))
    k_Cross = KFold(n_splits=3, random_state=0, shuffle=True)
    index = 0
    score = np.array([])
    data,label = dataset[:,:-1],dataset[:,-1]
    for train_index, test_index in k_Cross.split(dataset):
        train_data, train_label = data[train_index, :], label[train_index]
        test_data, test_label = data[test_index, :], label[test_index]
        model = StackingClassifier(
            classifiers=[
                KNeighborsClassifier(n_neighbors=2),
                Perceptron(eta0=0.01, max_iter=5000)
            ],
            meta_classifier=LogisticRegression()
        )
        model.fit(train_data,train_label)
        pred = model.predict(test_data)
        acc = accuracy_score(test_label, pred)
        score = np.append(score,acc)
        print('score[{}] = {}%'.format(index,acc))
        index+=1
    print('mean_accuracy = {}%'.format(np.mean(score)))
```


输出结果如下：

```python
score[0] = 0.8857142857142857%
score[1] = 0.7971014492753623%
score[2] = 0.8405797101449275%
mean_accuracy = 0.8411318150448585%
```
