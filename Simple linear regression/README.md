# Simple Linear Regression

> ​		线性回归是一种已有200多年历史的预测方法。简单线性回归是一种可以由训练集来估计属性的机器学习算法，它相对简单，便于初学者理解。在本节中，您将看到如何用C语言一步步实现这个算法。

## 1.算法介绍

​		线性回归可以在输入变量（X）与输出变量（Y）之间建立一种线性关系。具体的说，输出变量（Y）可以由输入变量（X）的线性组合计算出来。当输入变量为单一变量时，这种算法就叫做简单线性回归。

​		在简单线性回归中，我们可以使用训练数据的统计量来估计模型对新数据预测所需的系数。

一个简单线性回归模型可以写成:

$$
y = b_0 + b_1 × x\tag{1.1}
$$
​		其中，B0和B1为我们需要从训练集中估计的系数。得到系数后，我们可以利用此方程估计新的输入变量（X对应的输出变量（Y）。估计系数前，我们需要计算训练集的一些统计量，如平均值、方差和协方差等。

​		当计算出所需的统计量后，我们可以通过如下公式计算B0，B1：
$$
B_1 = \frac{\sum_{i=1}^{n}{((x_i - mean(x))×(y_i - mean(y)))}}{\sum_{i=1}^{n}{(x_i - mean(x))^2}}\tag{1.2}
$$

$$
B_0 = mean(y) - B_1 × mean(x)\tag{1.3}
$$

​		其中，i表示训练集中的第i个输入变量x或输出变量y。

## 2.算法实现步骤

本教程将通过以下八个部分实现简单线性回归算法：

- 读取csv
- 计算均值、方差及协协方差
- 估计回归系数
- 由回归系数计算测试集的预测值 
- 按划分的k折交叉验证计算预测所得准确率
- 按划分的训练集和测试集计算预测所得RMSE

  其中，读取csv、划分数据为k折、按划分的k折交叉验证计算预测所得准确率参考以前章节。

### 2.1 计算均值方差等统计量

- 功能——通过多个函数分别计算均值、方差及协方差
- 最终输出三个统计量的结果

#### 2.1.1 计算均值

​		输入变量（X）与输出变量（Y）的均值可以由以下公式得到：
$$
mean(x) = \frac{\sum_{i=1}{x_i}}{count(x)}\tag{1.4}
$$
​		其中，count(x)表示x的个数。

​		以下 mean() 函数可以计算一组数据的均值，它需要一维数组、数组长度作为参数。

**mean.c**

- ```C
  float mean(float* values, int length) {//对一维数组求均值
  	int i;
  	float sum = 0.0;
  	for (i = 0; i < length; i++) {
  		sum += values[i];
  	}
  	float mean = (float)(sum / length);
  	return mean;
  }
  ```

#### 2.1.2 计算方差

​		方差是每个值与均值之差的平方和。一组数字的方差可计算为:
$$
variance = \sum_{i=1}^{n}{(x_i - mean(x))^2}\tag{1.5}
$$
​		以下 variance()函数可以计算一组数据的方差，它需要一维数组变量、数组的均值、以及输出数组的长度作为参数。

** varianc.c**

- ```C
  float variance(float* values, float mean, int length) {//这里求的是平方和，没有除以n
  	float sum = 0.0;
  	int i;
  	for (i = 0; i < length; i++) {
  		sum += (values[i] - mean)*(values[i] - mean);
  	}
  	return sum;
  }
  ```

我们利用以下数据集：

```
float x[5]={1,2,4,3,5};
printf("%f\n",mean(x, 5));
printf("%f",variance(x,mean(x,5),5));
```

得到结果如下:

```
3.000 
10.000
```

#### 2.1.3 计算协方差

​		协方差在概率论和统计学中用于衡量两个变量的总体误差。而方差是协方差的一种特殊情况，即当两个变量是相同的情况。

​		协方差表示的是两个变量的总体的误差，这与只表示一个变量误差的方差不同。 如果两个变量的变化趋势一致，那么两个变量之间的协方差就是正值。 如果两个变量的变化趋势相反，那么两个变量之间的协方差就是负值。

​		我们可以通过以下公式来计算两个变量的协方差：
$$
covariance = {\sum_{i=1}^{n}{((x_i - mean(x))}}×(y_i - mean(y)))\tag{1.6}
$$
​		以下covariance()函数可以计算两组数据的协方差，它需要输入数组变量（X）、输入数组的均值、输出数组变量（Y）、输出数组的均值、数组长度作为参数。

**covariance.c**

- ```C
  float covariance(float* x, float mean_x, float* y, float mean_y, int length) {
  	float cov = 0.0;
  	int i = 0;
  	for (i = 0; i < length; i++) {
  		cov += (x[i] - mean_x)*(y[i] - mean_y);
  	}
  	return cov;
  } 
  ```

我们利用以下数据：

```
float x[5]={1,2,4,3,5};
float y[5]={1,3,3,2,5};
printf("%f",covariance(x,mean(x,5),y,mean(y,5),5));
```

得到如下结果：

```
8.000
```

### 2.2 估计回归系数

​		在简单线性回归中，我们需要估计两个系数的值。第一个是B1，可以利用公式(1.2)估计。

​		我们可以简化这个公式：
$$
B_1 = \frac{covariance(x,y)}{variance(x)}\tag{1.7}
$$
​		我们已经有了计算协方差和方差的函数。接下来，我们需要估计B0的值，也称为截距。可以利用公式(1.3)。

​		以下coefficients()函数将计算B0、B1并将其存在名为coef的数组。它需要训练集（二维数组），存储B0、B1的数组以及训练集数组长度作为参数。

- 功能——由均值方差估计回归系数
- 最终输出储存B0、B1值的数组

**coefficients.c**

```C
//由均值方差估计回归系数
void coefficients(float** data, float* coef, int length) {
	float* x = (float*)malloc(length * sizeof(float));
	float* y = (float*)malloc(length * sizeof(float));
	int i;
	for (i = 0; i < length; i++) {
		x[i] = data[i][0];
		y[i] = data[i][1];
		//printf("x[%d]=%f,y[%d]=%f\n",i, x[i],i,y[i]);
	}
	float x_mean = mean(x, length);
	float y_mean = mean(y, length);
	//printf("x_mean=%f,y_mean=%f\n", x_mean, y_mean);
	coef[1] = covariance(x, x_mean, y, y_mean, length) / variance(x, x_mean, length);
	coef[0] = y_mean - coef[1] * x_mean;
	for (i = 0; i < 2; i++) {
		printf("coef[%d]=%f\n", i, coef[i]);
	}
}
```



我们利用如下数据：

```
float data[3][2] = {
    {1,1},
    {2,2},
    {3,3}
};
float coef[2] = {1,1};
float* dataptr[3];
dataptr[0] = data[0];
dataptr[1] = data[1];
dataptr[2] = data[2];
coefficients(dataptr,coef,3);
```

coef作为输入的数组，经过函数操作后输出得到如下结果:

```
Ccoef[0]=0.000000
coef[1]=1.000000
```



### 2.3 计算测试集的预测值 

​		简单线性回归模型是一条由训练数据估计的系数定义的直线。系数估计出来后，我们就可以用它们来进行预测。用简单的线性回归模型进行预测的方程为公式(1.1)。

- 功能——由得到的线性回归模型计算测试的预测值
- 最终输出预测结果（Y）

** get_test_prediction.c**

- ``` C
  float* get_test_prediction(int col,int row,float** train, float** test, int n_folds) {
  	float* coef = (float*)malloc(col * sizeof(float));
  	int i;
  	for (i = 0; i < col; i++) {
  		coef[i] = 0.0;
  	}
  	int fold_size = (int)row / n_folds;
  	int train_size = fold_size * (n_folds - 1);
  	coefficients(train, coef, train_size);
  	float* predictions = (float*)malloc(fold_size * sizeof(float));
  	for (i = 0; i < fold_size; i++) {
  		predictions[i] = coef[0] + coef[1] * test[i][0];
  	}
  	return predictions;
  }
  ```



## 3. 完整算法及示例展示

​		我们现在知道如何实现一个简单的线性回归模型。我们把它应用到瑞典保险数据集。本节假设您已经将数据集下载到文件insurance.csv，并且它在当前工作目录中可用。。

​		下面列出了完整的示例。使用60%的训练数据集来准备模型，对剩下的40%进行预测

```C
#include<stdlib.h>
#include<stdio.h>

double **dataset;
int row, col;

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);

extern float* evaluate_algorithm(double **dataset, int row, int col, int n_folds);
float mean(float* values, int length);
float covariance(float* x, float mean_x, float* y, float mean_y, int length);
float variance(float* values, float mean, int length);
void coefficients(float** data, float* coef, int length);


int main() {	
	char filename[] = "Auto insurance.csv";
	char line[1024];
	row = get_row(filename);
	col = get_col(filename);
	dataset = (double **)malloc(row * sizeof(double *));
	for (int i = 0; i < row; ++i) {
		dataset[i] = (double *)malloc(col * sizeof(double));
	}//动态申请二维数组	
	get_two_dimension(line, dataset, filename);
	int n_folds = 10;
	int fold_size = (int)row / n_folds;
	evaluate_algorithm(dataset, row, col, n_folds);
	return 0;
}
//计算均值方差等统计量（多个函数）
float mean(float* values, int length) {//对一维数组求均值
	int i;
	float sum = 0.0;
	for (i = 0; i < length; i++) {
		sum += values[i];
	}
	float mean = (float)(sum / length);
	return mean;
}
float covariance(float* x, float mean_x, float* y, float mean_y, int length) {
	float cov = 0.0;
	int i = 0;
	for (i = 0; i < length; i++) {
		cov += (x[i] - mean_x)*(y[i] - mean_y);
	}
	return cov;
}
float variance(float* values, float mean, int length) {//这里求的是平方和，没有除以n
	float sum = 0.0;
	int i;
	for (i = 0; i < length; i++) {
		sum += (values[i] - mean)*(values[i] - mean);
	}
	return sum;
}
//由均值方差估计回归系数
void coefficients(float** data, float* coef, int length) {
	float* x = (float*)malloc(length * sizeof(float));
	float* y = (float*)malloc(length * sizeof(float));
	int i;
	for (i = 0; i < length; i++) {
		x[i] = data[i][0];
		y[i] = data[i][1];
		//printf("x[%d]=%f,y[%d]=%f\n",i, x[i],i,y[i]);
	}
	float x_mean = mean(x, length);
	float y_mean = mean(y, length);
	//printf("x_mean=%f,y_mean=%f\n", x_mean, y_mean);
	coef[1] = covariance(x, x_mean, y, y_mean, length) / variance(x, x_mean, length);
	coef[0] = y_mean - coef[1] * x_mean;
	/*for (i = 0; i < 2; i++) {
		printf("coef[%d]=%f\n", i, coef[i]);
	}*/
}


```





