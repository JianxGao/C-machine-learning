# **Multivariate Linear Regression**

>​		许多机器学习算法的核心是优化。优化算法通常被用来从给定的数据集中寻找一个良好的模型参数。机器学习中最常用的优化算法是随机梯度下降法。在本教程中，您将了解如何使用C语言从头开始实现随机梯度下降，并以此来优化线性回归算法。



## 1.算法介绍

### 1.1 多元线性回归

​		线性回归是一种预测真实值的算法。这些需要预测真实值的问题叫做回归问题。线性回归是一种使用直线来建立输入值和输出值之间关系模型的算法。在二维以上的空间中，这条直线被称为一个平面或超平面。

​		预测就是通过输入值的组合来预测输出值。每个输入属性(x)都使用一个系数(b)对其进行加权，学习算法的目标遍是寻找这组能产生良好预测(y)的系数。
$$
y = b_0+b_1×x_1+b_2×x_2+...\tag{1.1}
$$
​		这组系数可以用随机梯度下降法求得。

### 1.2 随机梯度下降

​		梯度下降是沿着函数的斜率或梯度最小化函数的过程。在机器学习中，我们可以使用一种算法来评估和更新每次迭代的效率，以最小化我们的模型在训练数据上的误差，这种算法被称为随机梯度下降法。

​		这种优化算法的工作原理是每次向模型提供一个训练样本，模型对这个训练样本进行预测，计算误差并更新模型以减少下一次预测的误差。此迭代过程将重复进行固定次数。

​		当我们得到某个模型的一个系数后，以下方程可以用来由已知系数计算新系数，从而使模型在训练数据上的误差最小。每次迭代，该方程都会使模型中的系数(b)更新。
$$
b = b-learning\space rate × error × x\tag{1.2}
$$
​		其中，b是被优化的系数或权值，学习率是你需要设定的参数(例如0.01)，error是模型由于权值问题导致在训练集上的预测误差，x是输入变量。

## 2.算法实现步骤

本教程将通过以下七个部分实现简单线性回归算法：

- 读取csv
- 归一化数据
- 划分数据为k折
- 训练集梯度下降估计回归系数
- 由回归系数计算预测值
- 计算测试集的预测值
- 计算RMSE
- 按划分的k折交叉验证计算预测所得准确率
- main函数设定以上函数的参数并调用

​        其中，读取csv、归一化数据、将数据划分为k折、计算测试集的预测值、计算RMSE、按划分的k折交叉验证计算预测所得准确率所用函数在以前章节已经介绍，在本节将着重介绍剩余的几个步骤。



### 2.1 训练集梯度下降估计回归系数

我们可以使用随机梯度下降法来估计训练数据的系数。随机梯度下降法需要两个参数：

- **Learning Rate**:用于限制每次更新时每个系数的修正量。
- **Epochs**:更新系数时遍历训练数据的次数。

以上参数及训练数据将作为函数的输入参数。我们需要在函数中执行以下3个循环:

- 循环每个Epoch。
- 对于每个Epoch，循环训练数据中的每一行。
- 对于每个Epoch中的每一行，循环并更新每个系数。

​        我们可以看到，对于每个Epochs，我们更新了训练数据中每一行的系数。这种更新是根据模型的误差产生的。其中，误差是用候选系数的预测值与预期输出值的差来计算的。
$$
error = prediction-expected\tag{1.3}
$$
​		每个输入属性都有一个权重系数，并且以相同的方式更新，例如:
$$
b_1(t+1) = b_1(t) - learning\space rate × error(t) × x_1(t)\tag{1.4}
$$
​		迭代最初的系数，也称为截距或偏置，同样以类似的方式更新。最终结果与系数的初始输入值无关。
$$
b_0(t+1) = b_0(t) - learning\space rate × error(t)\tag{1.5}
$$
下面是一个名为coefficients sgd()的函数，它使用随机梯度下降法计算训练数据集的系数。

- 功能——估计回归系数
- 以训练集数组、数组列数、系数存储数组、学习率、epoch、训练集长度作为输入参数。
- 最终输出系数存储数组。

**coefficients_sgd.c**

- ```c
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

```
weights[0] = 16140....
weights[1] = 56504....
weights[2] = 72645....
weights[3] = 88785....
weights[4] = 10492....
```



### 2.2 由回归系数计算预测值

​		在随机梯度下降中评估候选系数值时，在模型完成并且我们想要开始对测试数据或新数据进行预测时，我们都需要一个预测函数。

​		例如，有一个输入值(x)和两个系数值(b0和b1)。这个问题建模的预测方程为:
$$
y = b_0+b_1×x\tag{1.6}
$$
以下是一个名为predict()的预测函数，给定系数后，它可以预测一组输入值(x)对应的输出值(y)。

- 功能——预测输出值(y)
- 以输入值的属性个数、输入值、系数数组为输入参数
- 最终输出预测值

**predict.c**

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
27.000
```



## 3.完整算法及示例展示

​	在这一节中，我们将使用随机梯度下降法训练一个线性回归模型。利用的数据集为葡萄酒质量。该示例假设数据集的CSV副本位于当前工作目录中，文件名为winequize -white. CSV

​		我们将使用k-fold交叉验证来评估学习模型在未见数据上的性能。这意味着我们将构建和评估k个模型，并以平均模型误差来估计性能。均方根误差将用于评估每个模型。

​	我们将使用上面创建的predict()、coefficients_sgd()和线性回归sgd()函数来训练模型。下面是完整的示例。

```c
#include<stdlib.h>
#include<stdio.h>

double **dataset;
int row, col;

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern double* evaluate_algorithm(double **dataset, int row, int col, int n_folds, int n_epoch, double l_rate);
extern void normalize_dataset(double **dataset, int row, int col);

double* coefficients_sgd(double** dataset, int col, double coef[], double l_rate, int n_epoch, int train_size);
double predict(int col, double array[], double coefficients[]);
int main() {
	char filename[] = "winequality-white.csv";
	char line[1024];
	row = get_row(filename);
	col = get_col(filename);
	dataset = (double **)malloc(row * sizeof(double *));
	for (int i = 0; i < row; ++i) {
		dataset[i] = (double *)malloc(col * sizeof(double));
	}//动态申请二维数组	
	get_two_dimension(line, dataset, filename);
	normalize_dataset(dataset,row, col);
	
	int n_folds = 10;		
	double l_rate = 0.001f;
	int n_epoch = 50;	
	evaluate_algorithm(dataset, row,  col,  n_folds,  n_epoch, l_rate);
	return 0;
}


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
	/*for (i = 0; i < column; i++) {
		printf("coef[i]=%f\n", coef[i]);
	}*/
	return coef;
}

double predict(int col,double array[], double coefficients[]) {//预测某一行的值
	double yhat = coefficients[0];
	int i;
	for (i = 0; i < col - 1; i++)
		yhat += coefficients[i + 1] * array[i];
	return yhat;
}
```



