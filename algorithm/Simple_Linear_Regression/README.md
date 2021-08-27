## 3.1 Simple Linear Regression

> 线性回归是一种已有200多年历史的预测方法。简单线性回归是一种可以由训练集来估计属性的机器学习算法，它相对简单，便于初学者理解。在本节中，您将看到如何用C语言一步步实现这个算法。

### 3.1.1 算法介绍

线性回归可以在输入变量（X）与输出变量（Y）之间建立一种线性关系。具体的说，输出变量（Y）可以由输入变量（X）的线性组合计算出来。当输入变量为单一变量时，这种算法就叫做简单线性回归。

在简单线性回归中，我们可以使用训练数据的统计量来估计模型对新数据预测所需的系数。

一个简单线性回归模型可以写成:

$$
y = b_0 + b_1 × x\tag{1.1}
$$

其中，B0和B1为我们需要从训练集中估计的系数。得到系数后，我们可以利用此方程估计新的输入变量（X对应的输出变量（Y）。估计系数前，我们需要计算训练集的一些统计量，如平均值、方差和协方差等。

当计算出所需的统计量后，我们可以通过如下公式计算B0，B1：

$$
B_1 = \frac{\sum_{i=1}^{n}{((x_i - mean(x))×(y_i - mean(y)))}}{\sum_{i=1}^{n}{(x_i - mean(x))^2}}\tag{1.2}
$$

$$
B_0 = mean(y) - B_1 × mean(x)\tag{1.3}
$$

其中，i表示训练集中的第i个输入变量x或输出变量y。

### 3.1.2 算法讲解

本小节中，我们将通过代码来讲解算法。实现简单线性回归算法的步骤可以分为如下7个部分：

- 读取csv
- 计算均值、方差及协协方差
- 估计回归系数
- 由回归系数计算测试集的预测值
- 按划分的k折交叉验证计算预测所得准确率
- 按划分的训练集和测试集计算预测所得RMSE
- main函数设定以上函数的参数并调用

其中，读取csv、划分数据为k折、按划分的k折交叉验证计算预测所得准确率参考以前章节。

#### 计算均值方差等统计量

##### 计算均值

输入变量（X）与输出变量（Y）的均值可以由以下公式得到：

$$
mean(x) = \frac{\sum_{i=1}{x_i}}{count(x)}\tag{1.4}
$$

其中，`count(x)`表示x的个数。

以下 `mean()`函数可以计算一组数据的均值，它需要一维数组、数组长度作为参数。

```C
double mean(double* values, int length) 
{
    //对一维数组求均值
    int i;
    double sum = 0.0;
    for (i = 0; i < length; i++) {
        sum += values[i];
    }
    double mean = (double)(sum / length);
    return mean;
}
```

##### 计算方差

方差是每个值与均值之差的平方和。一组数字的方差可计算为:

$$
variance = \sum_{i=1}^{n}{(x_i - mean(x))^2}\tag{1.5}
$$

以下 `variance()`函数可以计算一组数据的方差，它需要一维数组变量、数组的均值、以及输出数组的长度作为参数。

```C
double variance(double* values, double mean, int length) {
    //这里求的是平方和，没有除以n
    double sum = 0.0;
    int i;
    for (i = 0; i < length; i++) {
        sum += (values[i] - mean)*(values[i] - mean);
    }
    return sum;
}
```

我们利用以下数据集：

```c
double x[5] = {1, 2, 4, 3, 5};
printf("%f", mean(x, 5));
printf("%f", variance(x, mean(x, 5), 5))
```

得到结果如下:

```c
3.000 
10.000
```

##### 计算协方差

协方差在概率论和统计学中用于衡量两个变量的总体误差。而方差是协方差的一种特殊情况，即当两个变量是相同的情况。

协方差表示的是两个变量的总体的误差，这与只表示一个变量误差的方差不同。 如果两个变量的变化趋势一致，那么两个变量之间的协方差就是正值。 如果两个变量的变化趋势相反，那么两个变量之间的协方差就是负值。

我们可以通过以下公式来计算两个变量的协方差：

$$
covariance = {\sum_{i=1}^{n}{((x_i - mean(x))}}×(y_i - mean(y)))\tag{1.6}
$$

以下 `covariance()`函数可以计算两组数据的协方差，它需要输入数组变量（X）、输入数组的均值、输出数组变量（Y）、输出数组的均值、数组长度作为参数。

```C
double covariance(double* x, double mean_x, double* y, double mean_y, int length) {
    double cov = 0.0;
    int i = 0;
    for (i = 0; i < length; i++) {
        cov += (x[i] - mean_x)*(y[i] - mean_y);
    }
    return cov;
} 
```

我们利用以下数据：

```c
double x[5] = {1, 2, 4, 3, 5};
double y[5] = {1, 3, 3, 2, 5};
printf("%f", covariance(x, mean(x, 5), y, mean(y, 5), 5));
```

得到如下结果：

```c
8.000
```

#### 估计回归系数

在简单线性回归中，我们需要估计两个系数的值。第一个是B1，可以利用公式(1.2)估计。

我们可以简化这个公式：

$$
B_1 = \frac{covariance(x,y)}{variance(x)}\tag{1.7}
$$

我们已经有了计算协方差和方差的函数。接下来，我们需要估计B0的值，也称为截距。可以利用公式(1.3)。

以下 `coefficients()`函数将计算B0、B1并将其存在名为coef的数组。它需要训练集（二维数组），存储B0、B1的数组以及训练集数组长度作为参数。

```C
//由均值方差估计回归系数
void coefficients(double** data, double* coef, int length) {
	double* x = (double*)malloc(length * sizeof(double));
	double* y = (double*)malloc(length * sizeof(double));
	int i;
	for (i = 0; i < length; i++) {
        x[i] = data[i][0];
        y[i] = data[i][1];
	}
	double x_mean = mean(x, length);
	double y_mean = mean(y, length);
	coef[1] = covariance(x, x_mean, y, y_mean, length) / variance(x, x_mean, length);
	coef[0] = y_mean - coef[1] * x_mean;
	for (i = 0; i < 2; i++) {
		printf("coef[%d]=%f\n", i, coef[i]);
	}
}
```

我们利用如下数据：

```c
double data[3][2] = {
    {1,1},
    {2,2},
    {3,3}
};
double coef[2] = {1,1};
double* dataptr[3];
dataptr[0] = data[0];
dataptr[1] = data[1];
dataptr[2] = data[2];
coefficients(dataptr, coef, 3);
```

coef作为输入的数组，经过函数操作后输出得到如下结果:

```c
Ccoef[0] = 0.000000
coef[1] = 1.000000
```

#### 计算测试集的预测值

简单线性回归模型是一条由训练数据估计的系数定义的直线。系数估计出来后，我们就可以用它们来进行预测。用简单的线性回归模型进行预测的方程为公式(1.1)。

以下 `get_test_prediction()`函数实现了对数据集的预测，它需要数据行数、列数、训练集、测试集、K折交叉验证数组大小作为输入参数。

```C
double* get_test_prediction(int col,int row,double** train, double** test, int n_folds) {
	double* coef = (double*)malloc(col * sizeof(double));
	int i;
	for (i = 0; i < col; i++) {
		coef[i] = 0.0;
	}
	int fold_size = (int)row / n_folds;
	int train_size = fold_size * (n_folds - 1);
	coefficients(train, coef, train_size);
	double* predictions = (double*)malloc(fold_size * sizeof(double));
	for (i = 0; i < fold_size; i++) {
		predictions[i] = coef[0] + coef[1] * test[i][0];
	}
	return predictions;
}
```

#### 计算预测准确率

将数据集按划分的k折交叉验证计算预测所得准确率，以下 `evaluate_algorithm()`函数需要训练集、测试集的二维数组，学习率，epoch数，交叉验证折数，交叉验证fold的长度作为参数输入。

```c
double evaluate_algorithm(double **dataset, int n_folds, int fold_size, double l_rate, int n_epoch) 
{
    double*** split =  cross_validation_split(double **dataset, int row, int n_folds, int fold_size);
    int i, j, k, l;
    int test_size = fold_size;
    int train_size = fold_size * (n_folds - 1); //train_size个一维数组
    double* score = (double*)malloc(n_folds * sizeof(float));
    for (i = 0; i < n_folds; i++) 
    {  
        //因为要遍历删除，所以拷贝一份split
        double*** split_copy = (double***)malloc(n_folds * sizeof(double**));
        for (j = 0; j < n_folds; j++) {
            split_copy[j] = (double**)malloc(fold_size * sizeof(double*));
            for (k = 0; k < fold_size; k++) {
                split_copy[j][k] = (double*)malloc(column * sizeof(double));
            }
        }
        for (j = 0; j < n_folds; j++)
        {
            for (k = 0; k < fold_size; k++)
            {
                for (l = 0; l < column; l++)
                {
                    split_copy[j][k][l] = split[j][k][l];
                }
            }
        }
        double** test_set = (double**)malloc(test_size * sizeof(double*));
        for (j = 0; j < test_size; j++) {//对test_size中的每一行
            test_set[j] = (double*)malloc(column * sizeof(double));
            for (k = 0; k < column; k++) {
                test_set[j][k] = split_copy[i][j][k];
            }
        }
        for (j = i; j < n_folds - 1; j++) {
            split_copy[j] = split_copy[j + 1];
        }
        double** train_set = (double**)malloc(train_size * sizeof(double*));
        for (k = 0; k < n_folds - 1; k++) {
            for (l = 0; l < fold_size; l++) {
                train_set[k*fold_size + l] = (double*)malloc(column * sizeof(double));
                train_set[k*fold_size + l] = split_copy[k][l];
            }
        }
        double* predicted = (double*)malloc(test_size * sizeof(double));//predicted有test_size个
        predicted = get_test_prediction(train_set, test_set, l_rate, n_epoch, fold_size);
        double* actual = (double*)malloc(test_size * sizeof(double));
        for (l = 0; l < test_size; l++) {
            actual[l] = test_set[l][column - 1];
        }
        double accuracy = accuracy_metric(actual, predicted, test_size);
        score[i] = accuracy;
        printf("score[%d]=%f\n", i, score[i]);
        free(split_copy);
    }
    double total = 0.0;
    for (l = 0; l < n_folds; l++) {
        total += score[l];
    }
    printf("mean_accuracy=%lf\%\n", total / n_folds);
    return score;
}
```

### 3.1.3 算法代码

我们现在知道了如何实现**简单线性回归算法**，那么我们把它应用到[瑞典保险数据集 insurance.csv](https://aistudio.baidu.com/aistudio/datasetdetail/105756/0)

我们给出链接：https://aistudio.baidu.com/aistudio/datasetdetail/105756/0

#### C语言细节讲解

本节假设您已下载数据集 `insurance.csv`，并且它在当前工作目录中可用。下面我们给出一个完整实例，使用C语言详细讲解每一处细节。我们给出每一个.c文件的所有代码：

##### 1) read_csv.c

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int get_row(char *filename) //获取行数
{
	char line[1024];
	int i = 0;
	FILE *stream = fopen(filename, "r");
	while (fgets(line, 1024, stream))
	{
		i++;
	}
	fclose(stream);
	return i;
}

int get_col(char *filename) //获取列数
{
	char line[1024];
	int i = 0;
	FILE *stream = fopen(filename, "r");
	fgets(line, 1024, stream);
	char *token = strtok(line, ",");
	while (token)
	{
		token = strtok(NULL, ",");
		i++;
	}
	fclose(stream);
	return i;
}

void get_two_dimension(char *line, double **data, char *filename)
{
	FILE *stream = fopen(filename, "r");
	int i = 0;
	while (fgets(line, 1024, stream)) //逐行读取
	{
		int j = 0;
		char *tok;
		char *tmp = strdup(line);
		for (tok = strtok(line, ","); tok && *tok; j++, tok = strtok(NULL, ",\n"))
		{
			data[i][j] = atof(tok); //转换成浮点数
		}							//字符串拆分操作
		i++;
		free(tmp);
	}
	fclose(stream); //文件打开后要进行关闭操作
}
```

##### 2) k_fold.c

```c
#include <stdlib.h>
#include <stdio.h>

double ***cross_validation_split(double **dataset, int row, int n_folds, int fold_size, int col)
{
	srand(10); //种子
	double ***split;
	int i, j = 0, k = 0;
	int index;
	double **fold;
	split = (double ***)malloc(n_folds * sizeof(double **));
	for (i = 0; i < n_folds; i++)
	{
		fold = (double **)malloc(fold_size * sizeof(double *));
		while (j < fold_size)
		{
			fold[j] = (double *)malloc(col * sizeof(double));
			index = rand() % row;
			fold[j] = dataset[index];
			for (k = index; k < row - 1; k++) //for循环删除这个数组中被rand取到的元素
			{
				dataset[k] = dataset[k + 1];
			}
			row--; //每次随机取出一个后总行数-1，保证不会重复取某一行
			j++;
		}
		j = 0; //清零j
		split[i] = fold;
	}
	return split;
}
```

##### 3) rmse.c

```c
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double rmse_metric(double *actual, double *predicted, int fold_size)
{
	double sum_err = 0.0;
	int i;
	for (i = 0; i < fold_size; i++)
	{
		double err = predicted[i] - actual[i];
		sum_err += err * err;
	}
	double mean_err = sum_err / fold_size;
	return sqrt(mean_err);
}
```

##### 4) test_prediction.c

```c
#include <stdlib.h>
#include <stdio.h>

extern void coefficients(double **data, double *coef, int length);
double *get_test_prediction(int col, int row, double **train, double **test, int n_folds)
{
	double *coef = (double *)malloc(col * sizeof(double));
	int i;
	for (i = 0; i < col; i++)
	{
		coef[i] = 0.0;
	}
	int fold_size = (int)row / n_folds;
	int train_size = fold_size * (n_folds - 1);
	coefficients(train, coef, train_size);
	double *predictions = (double *)malloc(fold_size * sizeof(double));
	for (i = 0; i < fold_size; i++)
	{
		predictions[i] = coef[0] + coef[1] * test[i][0];
	}
	return predictions;
}
```

##### 5) evaluate.c

```c
#include <stdlib.h>
#include <stdio.h>

extern double *get_test_prediction(int col, int row, double **train, double **test, int n_folds);
extern double rmse_metric(double *actual, double *predicted, int fold_size);
extern double ***cross_validation_split(double **dataset, int row, int col, int n_folds, int fold_size);

double *evaluate_algorithm(double **dataset, int row, int col, int n_folds)
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
				//printf("split_copy[%d][%d]=%f\n", k,l,split_copy[k][l]);
			}
		}

		double *predicted = (double *)malloc(test_size * sizeof(double));
		predicted = get_test_prediction(col, row, train_set, test_set, n_folds);
		double *actual = (double *)malloc(test_size * sizeof(double));
		for (l = 0; l < test_size; l++)
		{
			actual[l] = test_set[l][col - 1];
		}
		double rmse = rmse_metric(actual, predicted, test_size);
		score[i] = rmse;
		printf("score[%d] = %lf\n", i, score[i]);
		free(split_copy);
	}
	double total = 0;
	for (l = 0; l < n_folds; l++)
	{
		total += score[l];
	}
	printf("mean_rmse = %lf\n", total / n_folds);
	return score;
}
```

##### 6) main.c

```c
#include <stdlib.h>
#include <stdio.h>

double **dataset;
int row, col;

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern double *evaluate_algorithm(double **dataset, int row, int col, int n_folds);

double mean(double *values, int length);
double covariance(double *x, double mean_x, double *y, double mean_y, int length);
double variance(double *values, double mean, int length);
void coefficients(double **data, double *coef, int length);

//计算均值方差等统计量（多个函数）
double mean(double *values, int length)
{ //对一维数组求均值
	int i;
	double sum = 0.0;
	for (i = 0; i < length; i++)
	{
		sum += values[i];
	}
	double mean = (double)(sum / length);
	return mean;
}

double covariance(double *x, double mean_x, double *y, double mean_y, int length)
{
	double cov = 0.0;
	int i = 0;
	for (i = 0; i < length; i++)
	{
		cov += (x[i] - mean_x) * (y[i] - mean_y);
	}
	return cov;
}

double variance(double *values, double mean, int length)
{ //这里求的是平方和，没有除以n
	double sum = 0.0;
	int i;
	for (i = 0; i < length; i++)
	{
		sum += (values[i] - mean) * (values[i] - mean);
	}
	return sum;
}

//由均值方差估计回归系数
void coefficients(double **data, double *coef, int length)
{
	double *x = (double *)malloc(length * sizeof(double));
	double *y = (double *)malloc(length * sizeof(double));
	int i;
	for (i = 0; i < length; i++)
	{
		x[i] = data[i][0];
		y[i] = data[i][1];
	}
	double x_mean = mean(x, length);
	double y_mean = mean(y, length);
	coef[1] = covariance(x, x_mean, y, y_mean, length) / variance(x, x_mean, length);
	coef[0] = y_mean - coef[1] * x_mean;
}

int main()
{
	char filename[] = "Auto insurance.csv";
	char line[1024];
	row = get_row(filename);
	col = get_col(filename);
	dataset = (double **)malloc(row * sizeof(double *));
	for (int i = 0; i < row; ++i)
	{
		dataset[i] = (double *)malloc(col * sizeof(double));
	} //动态申请二维数组
	get_two_dimension(line, dataset, filename);
	int n_folds = 10;
	int fold_size = (int)row / n_folds;
	evaluate_algorithm(dataset, row, col, n_folds);
	return 0;
}
```

##### 7) compile.sh

```bash
gcc main.c read_csv.c k_fold.c evaluate.c rmse.c test_prediction.c -o run -lm && ./run
```

**编译&运行：**

```bash
bash compile.sh
```

最终输出结果如下：

```c
score[0] = 33.263512
score[1] = 30.319399
score[2] = 22.835829
score[3] = 38.080193
score[4] = 23.662033
score[5] = 25.166845
score[6] = 47.085342
score[7] = 46.614182
score[8] = 40.007351
score[9] = 28.511198
mean_rmse = 33.554588
```

#### Python语言实战

本节同样假设您已经下载数据集，我们使用著名机器学习开源库sklearn高效实现**简单线性回归算法**，以便您在实战中使用该算法：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


def rmse_metric(actual, predicted):
    sum_err = 0.0
    for i in range(len(actual)):
        err = predicted[i] - actual[i]
        sum_err += err ** 2
    mean_err = sum_err / (len(actual)-1)
    return np.sqrt(mean_err)


if __name__ == '__main__':
    dataset = np.array(pd.read_csv("insurance.csv", sep=',', header=None))
    k_Cross = KFold(n_splits=10, random_state=1, shuffle=True)
    index = 0
    score = np.array([])
    for train_index, test_index in k_Cross.split(dataset):
        train_data, train_label = dataset[train_index, :-1], dataset[train_index, -1]
        test_data, test_label = dataset[test_index, :-1], dataset[test_index, -1]
        model = LinearRegression()
        model.fit(train_data, train_label)
        pred = model.predict(test_data)
        rmse = rmse_metric(test_label,pred)
        score = np.append(score,rmse)
        print('score[{}] = {}'.format(index,rmse))
        index+=1
    print('mean_rmse = {}'.format(np.mean(score)))
```

输出结果如下：

```python
score[0] = 31.081476539821356
score[1] = 31.903964258437547
score[2] = 37.76453473731135
score[3] = 52.46285733249147
score[4] = 46.256226601172855
score[5] = 25.094234805956997
score[6] = 27.19738282646511
score[7] = 59.134038915742195
score[8] = 34.08824724550272
score[9] = 39.795062610664274
mean_rmse = 38.477802587356585
```

#### 