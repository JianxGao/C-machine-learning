# Perceptron

>​		感知器算法是一种最简单的人工神经网络。它是一个单神经元模型，可用于两类分类问题，并为以后开发更大的网络提供了基础。在本教程中，您将了解如何使用C语言从头开始实现感知器算法。

## 1.算法介绍

### 1.1 感知算法

​		感知器的灵感来自于单个神经细胞的信息处理过程，这种神经细胞被称为神经元。神经元通过树突接收输入信号，然后树突将电信号传递到细胞体。以类似的方式，感知器从训练数据的例子中接收输入信号，我们将其加权并组合成一个线性方程，这个方程被称为激活函数。
$$
activation = bias + \sum_{i=1}^{n}weight_i×x_i\tag{1.1}
$$
​		然后利用传递函数(如阶跃传递函数)将激活函数转化为输出值或预测值。
$$
prediction = 1.0 IF activation ≥ 0.0 ELSE 0.0\tag{1.2}
$$
​		这样，感知器就是一个解决两类问题（0和1）的分类算法，一个线性方程(如线或超平面)可以用来分离这两个类。它与线性回归和逻辑回归密切相关，后者以类似的方式进行预测(例如输入加权和)。感知器算法的权值必须使用随机梯度下降从训练数据中估计出来。

### 1.2 随机梯度下降

​		感知器算法使用梯度下降来更新权重。梯度下降在多元线性回归一节中进行了介绍和描述。每次梯度下降迭代，权值w根据公式更新如下:
$$
w = w + learning\space rate × (expected-predicted) × x\tag{1.3}
$$
​		其中，w是被优化的权重，learning rate是一个需要你配置的学习率（例如0.01），

(如0.01)，(expected - predicted)为模型对训练数据归属权值的预测误差，x为输入值。(expected - predicted)为模型对带有权值的训练数据的预测误差，x为输入值。

## 2.算法实现步骤

本教程将通过以下五个部分实现感知算法：

- 读取csv

- 划分数据为k折

- 训练集梯度下降估计回归系数

- 由回归系数计算预测值

- 按划分的k折交叉验证计算预测所得准确率

​        其中，读取csv、划分数据为k折、按划分的k折交叉验证计算预测所得准确率参考以前章节。

### 2.1 训练集梯度下降估计回归系数

我们可以使用随机梯度下降来估计训练数据的权值。随机梯度下降需要两个参数:

- **Learning Rate**:用于限制每个重量的数量，每次更新时修正。
- **Epochs**：更新权重时，遍历训练集的次数。

以上参数及训练数据将作为函数的参数。我们将在函数中执行3个循环：

- 每个Epoch循环一次
- 对于每一个Epoch，循环遍历训练集中的每一行
- 对于每一个Epoch中的每一行，循环遍历每个权重并更新它

​        如上，对于每一个Epoch，我们都更新了训练数据中每一行的每个权值。这个权值时根据模型产生的误差进行更新的。误差即期望输出值与由候选权值得到的预测值之间的差。

​		每个输入属性都有一个权重，并且以相同的方式更新这些权重。例如:
$$
w(t + 1) = w(t) + learning\space rate × (expected(t))- predicted(t)) × x(t)\tag{1.4}
$$
​		除了没有输入外，偏差也以类似的方式进行更新，因为它与特定的输入值没有关联:
$$
bias(t + 1) = bias(t) + learning\space rate × (expected(t))- predicted(t))\tag{1.5}
$$
​		现在我们可以把这些放在一起。下面是一个名为train weights()的函数，它使用的是随机梯度下降法来计算训练数据集的权值。

** train_weights.c**

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

```
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

```
weights[0] = 181.000
weights[1] = 322.000
weights[2] = 503.000
weights[3] = 684.000
weights[4] = 865.000
```



### 2.2 由回归系数计算预测值

​		开发一个可以进行预测的函数。这在随机梯度下降中评估候选权值时都需要用到，在模型完成后，我们希望对测试数据或新数据进行预测。下面是一个名为predict()的函数，它预测给定一组权重的行的输出值。第一个权重总是偏差，因为它是独立的，不对应任何特定的输入值。

**predict.c**

```c
double predict(int col,double *array, double *weights) {//预测某一行的值
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
    printf("%f",predict(5,data,weights));
}
```

得到如下结果：

```
1.0000
```



## 3.完整算法及示例展示

​		在这一节，我们将在声纳数据集上用随机梯度下降方法训练一个感知器模型。该示例假设数据集的CSV副本位于当前工作目录中，文件名为sonar.all-data.csv。首先加载数据集，将字符串值转换为数值，并将输出列从字符串转换为0到1的整数值。。

​		我们将使用k-fold交叉验证来评估学习模型在未见数据上的性能。这意味着我们将构建和评估k个模型，并以平均模型误差来估计性能。我们将用分类精度来评价每个模型。

​		我们将使用上面创建的predict()和train weights()函数来训练模型，并使用一个新的perceptron()函数将它们联系在一起。下面是完整的示例。

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

void train_weights(double **data, int col, double *weights, double l_rate, int n_epoch, int train_size);
double predict(int col, double *array, double *weights);
int main() {
	char filename[] = "sonar.csv";
	char line[1024];
	row = get_row(filename);
	col = get_col(filename);
	dataset = (double **)malloc(row * sizeof(double *));
	for (int i = 0; i < row; ++i) {
		dataset[i] = (double *)malloc(col * sizeof(double));
	}//动态申请二维数组	
	get_two_dimension(line, dataset, filename);
	normalize_dataset(dataset, row, col);

	int n_folds = 3;
	double l_rate = 0.01f;
	int n_epoch = 500;
	evaluate_algorithm(dataset, row, col, n_folds, n_epoch, l_rate);
	return 0;
}


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
}

double predict(int col,double *array, double *weights) {//预测某一行的值
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
