## 3.10 Backpropagation

### 3.10.1 算法介绍

反向传播算法（Backpropagation）是一种适合于多层神经元网络的学习算法，通常用于训练大规模的深度学习网络。反向传播算法主要基于梯度下降法，其过程由前向传播、反向传播、权重更新这三步构成。

下面将结合代码，详细阐述反向传播算法在MLP（多层感知机算法）中的应用过程。

### 3.10.2 算法讲解

#### 初始化网络参数

首先我们需要确定网络的层数与节点数，以本文为例，MLP的层数为两层，隐藏层（第一层）节点数为12，输出层（第二层）节点数为3：

```c
#define node1 12 //第一层节点数 
#define node2 3 //第二层节点数
```

之后定义sigmoid函数及其导数：

```c
//激活函数
double sigmoid(double x) 
{
    return 1.0 / (1.0 + exp(-x));
}

//激活函数的导数，y为激活函数值
double dsigmoid(double y)
{
    return y * (1.0 - y);  
}  
```

之后，根据层数和节点数初始化权重矩阵：

```c
double W_1[node1][col]; //第一层权重
double W_2[node2][node1+1]; //第二层权重 

// 初始化权重 
for(j=0;j<node1;j++){
    for(k=0;k<col;k++){
        W_1[j][k] = 0.1;
    }
}
for(j=0;j<node2;j++){
    for(k=0;k<node1+1;k++){
        W_2[j][k] = 0.1;
    }
}
```

将权重打印出来如下：

```
W_1[0][0] = 0.100000
W_1[0][1] = 0.100000
W_1[0][2] = 0.100000
W_1[0][3] = 0.100000
W_1[0][4] = 0.100000
W_1[0][5] = 0.100000
W_1[0][6] = 0.100000
W_1[0][7] = 0.100000
W_1[1][0] = 0.100000
W_1[1][1] = 0.100000
W_1[1][2] = 0.100000
W_1[1][3] = 0.100000
W_1[1][4] = 0.100000
W_1[1][5] = 0.100000
W_1[1][6] = 0.100000
W_1[1][7] = 0.100000
W_1[2][0] = 0.100000
W_1[2][1] = 0.100000
W_1[2][2] = 0.100000
W_1[2][3] = 0.100000
W_1[2][4] = 0.100000
W_1[2][5] = 0.100000
W_1[2][6] = 0.100000
W_1[2][7] = 0.100000
W_1[3][0] = 0.100000
W_1[3][1] = 0.100000
W_1[3][2] = 0.100000
W_1[3][3] = 0.100000
W_1[3][4] = 0.100000
W_1[3][5] = 0.100000
W_1[3][6] = 0.100000
W_1[3][7] = 0.100000
W_1[4][0] = 0.100000
W_1[4][1] = 0.100000
W_1[4][2] = 0.100000
W_1[4][3] = 0.100000
W_1[4][4] = 0.100000
W_1[4][5] = 0.100000
W_1[4][6] = 0.100000
W_1[4][7] = 0.100000
W_1[5][0] = 0.100000
W_1[5][1] = 0.100000
W_1[5][2] = 0.100000
W_1[5][3] = 0.100000
W_1[5][4] = 0.100000
W_1[5][5] = 0.100000
W_1[5][6] = 0.100000
W_1[5][7] = 0.100000
W_1[6][0] = 0.100000
W_1[6][1] = 0.100000
W_1[6][2] = 0.100000
W_1[6][3] = 0.100000
W_1[6][4] = 0.100000
W_1[6][5] = 0.100000
W_1[6][6] = 0.100000
W_1[6][7] = 0.100000
W_1[7][0] = 0.100000
W_1[7][1] = 0.100000
W_1[7][2] = 0.100000
W_1[7][3] = 0.100000
W_1[7][4] = 0.100000
W_1[7][5] = 0.100000
W_1[7][6] = 0.100000
W_1[7][7] = 0.100000
W_1[8][0] = 0.100000
W_1[8][1] = 0.100000
W_1[8][2] = 0.100000
W_1[8][3] = 0.100000
W_1[8][4] = 0.100000
W_1[8][5] = 0.100000
W_1[8][6] = 0.100000
W_1[8][7] = 0.100000
W_1[9][0] = 0.100000
W_1[9][1] = 0.100000
W_1[9][2] = 0.100000
W_1[9][3] = 0.100000
W_1[9][4] = 0.100000
W_1[9][5] = 0.100000
W_1[9][6] = 0.100000
W_1[9][7] = 0.100000
W_1[10][0] = 0.100000
W_1[10][1] = 0.100000
W_1[10][2] = 0.100000
W_1[10][3] = 0.100000
W_1[10][4] = 0.100000
W_1[10][5] = 0.100000
W_1[10][6] = 0.100000
W_1[10][7] = 0.100000
W_1[11][0] = 0.100000
W_1[11][1] = 0.100000
W_1[11][2] = 0.100000
W_1[11][3] = 0.100000
W_1[11][4] = 0.100000
W_1[11][5] = 0.100000
W_1[11][6] = 0.100000
W_1[11][7] = 0.100000
W_2[0][0] = 0.100000
W_2[0][1] = 0.100000
W_2[0][2] = 0.100000
W_2[0][3] = 0.100000
W_2[0][4] = 0.100000
W_2[0][5] = 0.100000
W_2[0][6] = 0.100000
W_2[0][7] = 0.100000
W_2[0][8] = 0.100000
W_2[0][9] = 0.100000
W_2[0][10] = 0.100000
W_2[0][11] = 0.100000
W_2[0][12] = 0.100000
W_2[1][0] = 0.100000
W_2[1][1] = 0.100000
W_2[1][2] = 0.100000
W_2[1][3] = 0.100000
W_2[1][4] = 0.100000
W_2[1][5] = 0.100000
W_2[1][6] = 0.100000
W_2[1][7] = 0.100000
W_2[1][8] = 0.100000
W_2[1][9] = 0.100000
W_2[1][10] = 0.100000
W_2[1][11] = 0.100000
W_2[1][12] = 0.100000
W_2[2][0] = 0.100000
W_2[2][1] = 0.100000
W_2[2][2] = 0.100000
W_2[2][3] = 0.100000
W_2[2][4] = 0.100000
W_2[2][5] = 0.100000
W_2[2][6] = 0.100000
W_2[2][7] = 0.100000
W_2[2][8] = 0.100000
W_2[2][9] = 0.100000
W_2[2][10] = 0.100000
W_2[2][11] = 0.100000
W_2[2][12] = 0.100000
```

#### 前向传播

权重初始化完成后，就可以开始进行训练。首先第一步是前向传播，设$a^{i}_j$为第$i$层第$j$个神经元的输出，$x_k$为输入向量的第$k$个元素，$w^{ij}_t$为第$i$层第$j$个神经元第$t$个权重值，计算出各个神经元的输出值：

$$
a^{1}_j = w^{1j}_0+\sum_{t}w^{1j}_{t}x_{t} \\
a^{i}_j = w^{ij}_0+\sum_{t}w^{ij}_{t}a^{i-1}_t \quad(i>1)
$$

代码片段如下：

```c
double layer1_out[node1]; //第一层节点输出值
double layer2_out[node2]; //第二层节点输出值
double train[7] = {15.26,14.84,0.871,5.763,3.312,2.221,5.22};

for(j=0;j<node1;j++){
    double sum = W_1[j][col-1];
    for(k=0;k<col-1;k++){
        sum += W_1[j][k]*train[k];
    }
    layer1_out[j] = sigmoid(sum);
    printf("layer1[%d] = %f\n",j,layer1_out[j]);
}

for(j=0;j<node2;j++){
    double sum = W_2[j][node1];
    for(k=0;k<node1;k++){
        sum += W_2[j][k]*layer1_out[k];
    }
    layer2_out[j] = sigmoid(sum);
    printf("layer2[%d] = %f\n",j,layer2_out[j]);
}
```

经计算后，我们可以得到如下结果：

```
layer1[0] = 0.992222
layer1[1] = 0.992222
layer1[2] = 0.992222
layer1[3] = 0.992222
layer1[4] = 0.992222
layer1[5] = 0.992222
layer1[6] = 0.992222
layer1[7] = 0.992222
layer1[8] = 0.992222
layer1[9] = 0.992222
layer1[10] = 0.992222
layer1[11] = 0.992222
layer2[0] = 0.784260
layer2[1] = 0.784260
layer2[2] = 0.784260
```

#### 反向传播

前向传播完成后，就可以计算预测值和真实值的误差，然后进行反向传播，为之后的权值更新做准备。设$\Delta a^{i}_j$为第$i$层第$j$个神经元的误差，$I$为总层数，实际值的第$j$个元素为$y_j$，$\delta$为激活函数的导数，则其公式为：

$$
\Delta a^{I}_j = (y_j-a^{I}_j)\delta(a^{I}_j) \\
\Delta a^{i}_t = \sum_{j}w^{(i+1)j}_{t}\delta(a^{i+1}_{j}) \quad (i<I)
$$

本文选取的例子为多分类问题，需要把真实值（1,2,3）转换成one-hot形式（[1,0,0],[0,1,0],[0,0,1]）。定义转换函数如下：

```c
#define class_num 3 //种类数量 

double *transfer_to_one_hot(int y){
    double *one_hot = (double *)malloc(class_num*sizeof(double));
    int i;
    for(i=0;i<class_num;i++){
        one_hot[i] = 0;
    }
    one_hot[y-1] = 1;
    return one_hot;
}
```

于是利用转换函数把原数据转换为one-hot形式，再计算误差进行反向传播，代码片段如下：

```c
// 误差反向传播
int y = 1;
double *target = transfer_to_one_hot(y);
double layer2_delta[node2];
double layer1_delta[node1];
for(j=0;j<node2;j++){
    double expected= (double) *(target + j);
    layer2_delta[j] = (expected - layer2_out[j])*dsigmoid(layer2_out[j]);
}
for(j=0;j<node1;j++){
    double error = 0.0;
    for(k=0;k<node2;k++){
        error += W_2[k][j]*layer2_delta[k];
    }
    layer1_delta[j] = error*dsigmoid(layer1_out[j]);
}
```

经过计算后，我们可以得到各个层的delta值，结果如下：

```
layer2_delta[0] = 0.036502
layer2_delta[1] = -0.132694
layer2_delta[2] = -0.132694
layer1_delta[0] = -0.000177
layer1_delta[1] = -0.000177
layer1_delta[2] = -0.000177
layer1_delta[3] = -0.000177
layer1_delta[4] = -0.000177
layer1_delta[5] = -0.000177
layer1_delta[6] = -0.000177
layer1_delta[7] = -0.000177
layer1_delta[8] = -0.000177
layer1_delta[9] = -0.000177
layer1_delta[10] = -0.000177
layer1_delta[11] = -0.000177
```

#### 权重更新

在得到各个神经元的差值($\Delta a$)后，就可以对原先的权重进行更新。设学习率为$r$，于是可以得到公式：

$$
w^{1j}_t = w^{1j}_t + rx_t\Delta a^{i}_{j} \quad (t>0) \\
w^{1j}_0 = w^{1j}_0 + r\Delta a^{i}_{j} \\
w^{ij}_t = w^{ij}_t + ra^{i}_{t}\Delta a^{i}_{j} \quad (t>0) \\
w^{ij}_0 = w^{ij}_t + r\Delta a^{i}_{j}
$$

写成代码片段如下：

```c
// 更新权重
double l_rate = 0.01;
for(j=0;j<node1;j++){
    for(k=0;k<col-1;k++){
        W_1[j][k] += l_rate*layer1_delta[j]*train[k];
        printf("W_1[%d][%d] = %f\n",j,k,W_1[j][k]);
    }
    W_1[j][col] += l_rate*layer1_delta[j];
    printf("W_1[%d][%d] = %f\n",j,col,W_1[j][col]);
}
for(j=0;j<node2;j++){
    for(k=0;k<node1+1;k++){
        W_2[j][k] += l_rate*layer2_delta[j]*layer1_out[k];
        printf("W_2[%d][%d] = %f\n",j,k,W_2[j][k]);
    }
    W_2[j][node1] += l_rate*layer2_delta[j];
    printf("W_2[%d][%d] = %f\n",j,col,W_2[j][col]);
}
```

把更新后的权重结果打印如下：

```
W_1[0][0] = 0.099973
W_1[0][1] = 0.099974
W_1[0][2] = 0.099998
W_1[0][3] = 0.099990
W_1[0][4] = 0.099994
W_1[0][5] = 0.099996
W_1[0][6] = 0.099991
W_1[0][8] = 0.099998
W_1[1][0] = 0.099971
W_1[1][1] = 0.099974
W_1[1][2] = 0.099998
W_1[1][3] = 0.099990
W_1[1][4] = 0.099994
W_1[1][5] = 0.099996
W_1[1][6] = 0.099991
W_1[1][8] = 0.099998
W_1[2][0] = 0.099971
W_1[2][1] = 0.099974
W_1[2][2] = 0.099998
W_1[2][3] = 0.099990
W_1[2][4] = 0.099994
W_1[2][5] = 0.099996
W_1[2][6] = 0.099991
W_1[2][8] = 0.099998
W_1[3][0] = 0.099971
W_1[3][1] = 0.099974
W_1[3][2] = 0.099998
W_1[3][3] = 0.099990
W_1[3][4] = 0.099994
W_1[3][5] = 0.099996
W_1[3][6] = 0.099991
W_1[3][8] = 0.099998
W_1[4][0] = 0.099971
W_1[4][1] = 0.099974
W_1[4][2] = 0.099998
W_1[4][3] = 0.099990
W_1[4][4] = 0.099994
W_1[4][5] = 0.099996
W_1[4][6] = 0.099991
W_1[4][8] = 0.099998
W_1[5][0] = 0.099971
W_1[5][1] = 0.099974
W_1[5][2] = 0.099998
W_1[5][3] = 0.099990
W_1[5][4] = 0.099994
W_1[5][5] = 0.099996
W_1[5][6] = 0.099991
W_1[5][8] = 0.099998
W_1[6][0] = 0.099971
W_1[6][1] = 0.099974
W_1[6][2] = 0.099998
W_1[6][3] = 0.099990
W_1[6][4] = 0.099994
W_1[6][5] = 0.099996
W_1[6][6] = 0.099991
W_1[6][8] = 0.099998
W_1[7][0] = 0.099971
W_1[7][1] = 0.099974
W_1[7][2] = 0.099998
W_1[7][3] = 0.099990
W_1[7][4] = 0.099994
W_1[7][5] = 0.099996
W_1[7][6] = 0.099991
W_1[7][8] = 0.099998
W_1[8][0] = 0.099971
W_1[8][1] = 0.099974
W_1[8][2] = 0.099998
W_1[8][3] = 0.099990
W_1[8][4] = 0.099994
W_1[8][5] = 0.099996
W_1[8][6] = 0.099991
W_1[8][8] = 0.099998
W_1[9][0] = 0.099971
W_1[9][1] = 0.099974
W_1[9][2] = 0.099998
W_1[9][3] = 0.099990
W_1[9][4] = 0.099994
W_1[9][5] = 0.099996
W_1[9][6] = 0.099991
W_1[9][8] = 0.099998
W_1[10][0] = 0.099971
W_1[10][1] = 0.099974
W_1[10][2] = 0.099998
W_1[10][3] = 0.099990
W_1[10][4] = 0.099994
W_1[10][5] = 0.099996
W_1[10][6] = 0.099991
W_1[10][8] = 0.099998
W_1[11][0] = 0.099971
W_1[11][1] = 0.099974
W_1[11][2] = 0.099998
W_1[11][3] = 0.099990
W_1[11][4] = 0.099994
W_1[11][5] = 0.099996
W_1[11][6] = 0.099991
W_1[11][8] = -0.000002
W_2[0][0] = 0.100362
W_2[0][1] = 0.100362
W_2[0][2] = 0.100362
W_2[0][3] = 0.100362
W_2[0][4] = 0.100362
W_2[0][5] = 0.100362
W_2[0][6] = 0.100362
W_2[0][7] = 0.100362
W_2[0][8] = 0.100362
W_2[0][9] = 0.100362
W_2[0][10] = 0.100362
W_2[0][11] = 0.100362
W_2[0][12] = 0.100037
W_2[0][8] = 0.100362
W_2[1][0] = 0.098683
W_2[1][1] = 0.098683
W_2[1][2] = 0.098683
W_2[1][3] = 0.098683
W_2[1][4] = 0.098683
W_2[1][5] = 0.098683
W_2[1][6] = 0.098683
W_2[1][7] = 0.098683
W_2[1][8] = 0.098683
W_2[1][9] = 0.098683
W_2[1][10] = 0.098683
W_2[1][11] = 0.098683
W_2[1][12] = 0.099867
W_2[1][8] = 0.098683
W_2[2][0] = 0.098683
W_2[2][1] = 0.098683
W_2[2][2] = 0.098683
W_2[2][3] = 0.098683
W_2[2][4] = 0.098683
W_2[2][5] = 0.098683
W_2[2][6] = 0.098683
W_2[2][7] = 0.098683
W_2[2][8] = 0.098683
W_2[2][9] = 0.098683
W_2[2][10] = 0.098683
W_2[2][11] = 0.098683
W_2[2][12] = 0.099867
W_2[2][8] = 0.098683
```

#### 预测

训练完成后，就可以利用训练好的权重矩阵进行预测。其过程和前向传播大致相同。代码如下：

```c
// 预测
double *predictions = (double *)malloc(test_size*sizeof(double));
for(i=0;i<test_size;i++){
    double out1[node1];
    for(j=0;j<node1;j++){
        out1[j] = W_1[j][col-1];
        for(k=0;k<col-1;k++){
            out1[j] += W_1[j][k]*test[i][k];
        }
        out1[j] = sigmoid(out1[j]);
    }
    double out2[node2];
    for(j=0;j<node2;j++){
        double max;
        out2[j] = W_2[j][node1];
        for(k=0;k<node1;k++){
            out2[j] += W_2[j][k]*out1[k];
        }
        out2[j] = sigmoid(out2[j]);
        if(j>0){
            if(out2[j]>max){
                predictions[i] = j+1;
                max = out2[j];
            }
        }else{
            predictions[i] = 1;
            max = out2[j];
        }
    }
}
```

### 3.10.3 算法代码

我们现在知道了如何实现**BP算法**，那么我们把它应用到[小麦种子数据集 seeds_data.csv](https://aistudio.baidu.com/aistudio/datasetdetail/105756/0)

我们给出链接：https://aistudio.baidu.com/aistudio/datasetdetail/105756/0

#### C语言细节讲解

本节假设您已下载数据集 `seeds_data.csv`，并且它在当前工作目录中可用。下面我们给出一个完整实例，使用C语言详细讲解每一处细节。我们给出每一个.c文件的所有代码：

##### 1) read_csv.c

该文件代码与前面代码一致，不再重复给出。

##### 2) normalize.c

该文件代码与前面代码一致，不再重复给出。

##### 3) k_fold.c

该文件代码与前面代码一致，不再重复给出。

##### 4) score.c

该文件代码与前面代码一致，不再重复给出。

##### 5) test_prediction.c

```c
#define randval(high) ((double)rand() / RAND_MAX * high)
#define uniform_plus_minus_one ((double)(2.0 * rand()) / ((double)RAND_MAX + 1.0) - 1.0) //均匀随机分布

#define node1 12	//第一层节点数
#define node2 3		//第二层节点数
#define class_num 3 //种类数量

#include "math.h"
#include "stdlib.h"
#include "time.h"
#include "assert.h"
#include "string.h"
#include "stdio.h"

//激活函数
double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

//激活函数的导数，y为激活函数值
double dsigmoid(double y)
{
	return y * (1.0 - y);
}

double *transfer_to_one_hot(int y)
{
	double *one_hot = (double *)malloc(class_num * sizeof(double));
	int i;
	for (i = 0; i < class_num; i++)
	{
		one_hot[i] = 0;
	}
	one_hot[y - 1] = 1;
	return one_hot;
}

//训练模型并获得预测值
double *get_test_prediction(double **train, double **test, double l_rate, int n_epoch, int train_size, int test_size, int col)
{
	int epoch, i, j, k;
	// 初始化权重
	double W_1[node1][col];		  //第一层权重
	double W_2[node2][node1 + 1]; //第二层权重
	double out;
	double predict;
	double layer1_out[node1]; //第一层节点输出值
	double layer2_out[node2]; //第二层节点输出值
	// 初始化权重
	for (j = 0; j < node1; j++)
	{
		for (k = 0; k < col; k++)
		{
			W_1[j][k] = uniform_plus_minus_one;
		}
	}
	for (j = 0; j < node2; j++)
	{
		for (k = 0; k < node1 + 1; k++)
		{
			W_2[j][k] = uniform_plus_minus_one;
		}
	}
	for (epoch = 0; epoch < n_epoch; epoch++)
	{
		for (i = 0; i < train_size; i++)
		{
			// 前向传播
			for (j = 0; j < node1; j++)
			{
				double sum = W_1[j][col - 1];
				for (k = 0; k < col - 1; k++)
				{
					sum += W_1[j][k] * train[i][k];
				}
				layer1_out[j] = sigmoid(sum);
			}
			for (j = 0; j < node2; j++)
			{
				double sum = W_2[j][node1];
				for (k = 0; k < node1; k++)
				{
					sum += W_2[j][k] * layer1_out[k];
				}
				layer2_out[j] = sigmoid(sum);
			}
			// 误差反向传播
			int y;
			y = (int)train[i][col - 1];
			double *target = transfer_to_one_hot(y);
			double layer2_delta[node2];
			double layer1_delta[node1];
			for (j = 0; j < node2; j++)
			{
				double expected = (double)*(target + j);
				layer2_delta[j] = (expected - layer2_out[j]) * dsigmoid(layer2_out[j]);
			}
			for (j = 0; j < node1; j++)
			{
				double error = 0.0;
				for (k = 0; k < node2; k++)
				{
					error += W_2[k][j] * layer2_delta[k];
				}
				layer1_delta[j] = error * dsigmoid(layer1_out[j]);
			}
			// 更新权重
			for (j = 0; j < node1; j++)
			{
				for (k = 0; k < col - 1; k++)
				{
					W_1[j][k] += l_rate * layer1_delta[j] * train[i][k];
				}
				W_1[j][col] += l_rate * layer1_delta[j];
			}
			for (j = 0; j < node2; j++)
			{
				for (k = 0; k < node1 + 1; k++)
				{
					W_2[j][k] += l_rate * layer2_delta[j] * layer1_out[k];
				}
				W_2[j][node1] += l_rate * layer2_delta[j];
			}
		}
	}
	// 预测
	double *predictions = (double *)malloc(test_size * sizeof(double));
	for (i = 0; i < test_size; i++)
	{
		double out1[node1];
		for (j = 0; j < node1; j++)
		{
			out1[j] = W_1[j][col - 1];
			for (k = 0; k < col - 1; k++)
			{
				out1[j] += W_1[j][k] * test[i][k];
			}
			out1[j] = sigmoid(out1[j]);
		}
		double out2[node2];
		for (j = 0; j < node2; j++)
		{
			double max;
			out2[j] = W_2[j][node1];
			for (k = 0; k < node1; k++)
			{
				out2[j] += W_2[j][k] * out1[k];
			}
			out2[j] = sigmoid(out2[j]);
			if (j > 0)
			{
				if (out2[j] > max)
				{
					predictions[i] = j + 1;
					max = out2[j];
				}
			}
			else
			{
				predictions[i] = 1;
				max = out2[j];
			}
		}
	}
	return predictions;
}
```

##### 6) evaluate.c

```c
#include <stdlib.h>
#include <stdio.h>

extern double ***cross_validation_split(double **dataset, int row, int n_folds, int fold_size, int col);
extern double *get_test_prediction(double **train, double **test, double l_rate, int n_epoch, int train_size, int test_size, int col);
extern double accuracy_metric(double *actual, double *predicted, int fold_size);

double *evaluate_algorithm(double **dataset, int n_folds, int fold_size, double l_rate, int n_epoch, int col, int row)
{
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
		double *predicted_2;
		predicted_2 = get_test_prediction(train_set, test_set, l_rate, n_epoch, train_size, test_size, col);
		double predicted[test_size];
		double *actual = (double *)malloc(test_size * sizeof(double));
		for (l = 0; l < test_size; l++)
		{
			predicted[l] = (double)*(predicted_2 + l);
			actual[l] = test_set[l][col - 1];
		}
		double accuracy = accuracy_metric(actual, predicted, test_size);
		score[i] = accuracy;
		printf("score[%d]=%f\n", i, score[i]);
		free(split_copy);
	}
	double total = 0.0;
	for (l = 0; l < n_folds; l++)
	{
		total += score[l];
	}
	printf("mean_accuracy=%f\n", total / n_folds);
	return score;
}
```

##### 7) main.c

```c
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern double *evaluate_algorithm(double **dataset, int n_folds, int fold_size, double l_rate, int n_epoch, int col, int row);

void main()
{
    char filename[] = "seeds_data.csv";
    char line[1024];
    int row = get_row(filename);
    int col = get_col(filename);
    printf("row = %d\n", row);
    printf("col = %d\n", col);
    double **dataset = (double **)malloc(row * sizeof(int *));
    int i;
    for (i = 0; i < row; ++i)
    {
        dataset[i] = (double *)malloc(col * sizeof(double));
    }
    get_two_dimension(line, dataset, filename);
    double l_rate = 0.1;
    int n_epoch = 100;
    int n_folds = 4;
    int fold_size;
    fold_size = (int)(row / n_folds);
    evaluate_algorithm(dataset, n_folds, fold_size, l_rate, n_epoch, col, row);
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

运算后得到的结果如下：

```c
score[0] = 95.918367%
score[1] = 83.673469%
score[2] = 79.591837%
score[3] = 57.142857%
mean_accuracy = 79.081633%
```

#### Python语言实战

本节同样假设您已经下载数据集，我们使用著名机器学习开源库sklearn高效实现**BP算法**，以便您在实战中使用该算法：

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    dataset = np.array(pd.read_csv("seeds_data.csv", sep=',', header=None))
    k_Cross = KFold(n_splits=5, random_state=8, shuffle=True)
    index = 0
    score = np.array([])
    Scaler = MinMaxScaler()
    data, label = dataset[:,:-1], dataset[:,-1]
    data = Scaler.fit_transform(data)
    for train_index, test_index in k_Cross.split(dataset):
        train_data, train_label = data[train_index, :], label[train_index]
        test_data, test_label = data[test_index, :], label[test_index]
        model = MLPClassifier(max_iter=100)
        model.fit(train_data, train_label)
        pred = model.predict(test_data)
        acc = accuracy_score(test_label, pred) * 100
        score = np.append(score,acc)
        print('score[{}] = {}%'.format(index,acc))
        index+=1
    print('mean_accuracy = {}%'.format(np.mean(score)))
```

输出结果如下：

```python
score[0] = 92.5%
score[1] = 90.0%
score[2] = 85.0%
score[3] = 82.5%
score[4] = 100.0%
mean_accuracy = 90.0%
```

