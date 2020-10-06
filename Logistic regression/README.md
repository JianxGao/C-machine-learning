# Logistic Regression

> ​		逻辑回归是一种针对两类问题的归一化线性分类算法。它易于实现，易于理解，并且在各种各样的问题上都能得到很好的结果。在本教程中，您将了解如何借助C语言，使用随机梯度下降从零开始实现逻辑回归

## 1. 算法介绍

### 1.1 逻辑回归

​		逻辑回归是根据该方法的核心函数——Logistic函数而命名的。逻辑回归可以使用一个方程来表示，与线性回归非常相似。输入值(X)加以权重值或系数值，然后通过它们的线性组合来预测输出值(y)。逻辑与线性回归的一个关键区别是，建模的输出值是二进制值(0或1)，而不是数值。
$$
yhat = \frac{e^{b0+b1×x1}}{1+e^{b0+b2×x1}}\tag{1.1}
$$
可以简化为：
$$
yhat = \frac{1.0}{1.0+e^{-(b0+b2×x1)}}\tag{1.2}
$$
其中e为自然对数(欧拉数)的底数，yhat为预测输出，b0为偏置或截距项，b1为单个输入值(x1)的系数。预测值ythat是一个0到1之间的实值，需要四舍五入到一个整数值并映射到一个预测的类值。

​		输入数据中的每一列都有一个相关的b系数(一个常实值)，它必须从训练数据中获得。您存储在内存或文件中的模型实际是方程中的系数(beta值或b)。Logistic回归算法的系数由训练数据估计而来。

### 1.2 随机梯度下降

​		Logistic回归使用梯度下降更新系数。梯度下降在**8.1.2**节中进行了介绍和描述。每次梯度下降迭代时，机器学习语言中的系数(b)根据以下公式更新:
$$
b = b+learning\space rate × (y-yhat)×yhat×(1-yhat)×x\tag{1.3}
$$
​		其中，b是将被优化的系数或权重，learning rate一个学习速率,它需要您的配置(例如0.01),(y - yhat)是模型在被分配有权重的训练集上的误差,yhat是由预测系数得到的预测值，x是输入值。

## 2. 算法实现步骤

本教程将通过以下五个部分实现逻辑回归算法：

- 读取csv

- 归一化数据

- 划分数据为k折

- 训练集梯度下降估计回归系数

- 由回归系数计算预测值

- 按划分的k折交叉验证计算预测所得准确率

  其中，读取csv、划分数据为k折、按划分的k折交叉验证计算预测所得准确率参考以前章节。

### 2.1 训练集梯度下降估计回归系数

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
​		列表开始处的特殊系数，也称为截距，以类似的方式更新，只是没有输入，因为它与特定的输入值没有关联:
$$
b0(t+1)=b0(t)+learning\space rate ×(y(t)-yhat(t)×yhat(t)×(1-yhat(t))\tag{1.4}
$$
​		现在我们可以把这些放在一起。下面是一个名为coefficients_sgd()的函数，它使用随机梯度下降法计算训练数据集的系数值。

**coefficients_sgd.c**

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

```
weights[0] = 1.000
weights[1] = 2.000
weights[2] = 3.000
weights[3] = 4.000
weights[4] = 5.000
```



### 2.2 由回归系数计算预测值

​		在随机梯度下降中评估候选系数值时，以及在模型完成并且我们希望开始对测试数据或新数据进行预测后，都需要一个可以预测的函数。下面是一个名为predict()的函数，给定一组系数后，它可以输出一行预测值。第一个系数是截距，也称为偏差或b0，因为它是独立的，不对应特定的输入值。.c

**predict.c**

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

```
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





## 3. 完整算法及示例展示

​		在本节中，我们将使用糖尿病数据集的随机梯度下降训练一个逻辑回归模型。该示例假设数据集的CSV副本位于当前工作目录中，文件名为pima-indians-diabetes.csv.

​		首先加载数据集，将字符串值转换为数值，并将每个列规范化为范围为0到1的值。

​		我们将使用k-fold交叉验证来评估学习模型在未知数据上的性能。这意味着我们将构建和评估k个模型，并将性能估计为模型的平均性能。用分类精度来评价每个模型。

​		我们将使用上面创建的predict()和coefficients_sgd()函数以及一个新的logistic_regression()函数来训练模型。下面是完整的示例。

```c
#include<stdlib.h>
#include<stdio.h>
#include<math.h>

double **dataset;
int row, col;

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern double* evaluate_algorithm(double **dataset, int row, int col, int n_folds, int n_epoch, double l_rate);
extern void normalize_dataset(double **dataset, int row, int col);

void coefficients_sgd(double ** dataset, int col, double *coef, double l_rate, int n_epoch, int train_size);
double predict(int col, double array[], double coefficients[]);
int main() {
	char filename[] = "Pima.csv";
	char line[1024];
	row = get_row(filename);
	col = get_col(filename);
	dataset = (double **)malloc(row * sizeof(double *));
	for (int i = 0; i < row; ++i) {
		dataset[i] = (double *)malloc(col * sizeof(double));
	}//动态申请二维数组	
	get_two_dimension(line, dataset, filename);
	normalize_dataset(dataset, row, col);

	int n_folds = 5;
	double l_rate = 0.1f;
	int n_epoch = 100;
	evaluate_algorithm(dataset, row, col, n_folds, n_epoch, l_rate);
	return 0;
}


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
		//printf("coef[%d]=%f\n",i, coef[i]);
	}
}

double predict(int col, double array[], double coefficients[]) {//预测某一行的值
	double yhat = coefficients[0];
	int i;
	for (i = 0; i < col - 1; i++)
		yhat += coefficients[i + 1] * array[i];
	return 1 / (1 + exp(-yhat));
}
```


