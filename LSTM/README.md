# LSTM算法

## 1.算法介绍

LSTM，长短期记忆网络，全称为Long Short Term Memory networks。它是基于RNN的一种时间循环神经网络。

在理解LSTM之前，首先需要了解循环神经网络（RNN）的原理。

### 1.1 RNN与LSTM

人的思维是连续的，思考问题并不会从头开始，而是会“结合上下文”。传统的神经网络并不能做到这点，而RNN正是这一问题的解决方案。

循环神经网络（RNN）中的神经元，可以把输出值作为下一个神经元的输入值的一部分，进而保证神经网络能够连续“思考”。

![img](https://upload-images.jianshu.io/upload_images/6983308-42172a6dae3d3388.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

然而RNN并不完美，它存在“长依赖”的问题。比方说，假设想让RNN根据一段不完整的句子来预测缺失的单词，例如“I grew up in France… I speak fluent ________.”（缺失的单词为French)，则有用的信息主要集中在前半句。然而要预测的单词却和前面有用的信息距离较远，这会导致RNN很难学习到有用的信息。

而LSTM解决了RNN的长依赖问题。如图所示，LSTM也是链状结构，但它和RNN的不同之处在于中间的神经元变成了一个较为复杂的细胞，其主要由遗忘门、输入门、输出门和记忆部分组成。而这个模块正是LSTM的核心。

![img](https://upload-images.jianshu.io/upload_images/6983308-2f0d4a87883d2c8c.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

## 2.算法实现步骤

下面将以LSTM学习加法为例，具体介绍如何使用C语言实现LSTM模型。

### 2.1  读取csv

该步骤代码与前面代码一致，不再重复给出。

### 2.2  划分数据为k折

该步骤代码与前面代码一致，不再重复给出。

### 2.3  核心算法

#### 2.3.1 初始化

首先我们需要确定LSTM的细胞数、输入结点数（$x_t$的维度）和隐藏结点数（$h_t$的维度）。以本节为例，设置输入结点数为2，隐藏结点数为12，细胞数为8。代码如下：

```c
#define innode  2       //输入结点数
#define hidenode  12   //隐藏结点数
#define cell_num 8  //LSTM细胞数 
```

再定义均匀随机分布：

```c
#define uniform_plus_minus_one ( (double)( 2.0 * rand() ) / ((double)RAND_MAX + 1.0) - 1.0 )  //均匀随机分布 
```

依照定义，我们可以初始化LSTM网络的权重矩阵：

```c
double W_I[innode][hidenode];     //连接输入与细胞输入门的权值矩阵
double U_I[hidenode][hidenode];   //连接上一细胞输出与本细胞单元中输入门的权值矩阵
double W_F[innode][hidenode];     //连接输入与细胞遗忘门的权值矩阵
double U_F[hidenode][hidenode];   //连接上一细胞与本细胞中遗忘门的权值矩阵
double W_O[innode][hidenode];     //连接输入与细胞输出门的权值矩阵
double U_O[hidenode][hidenode];   //连接上一细胞与现在时刻的细胞的权值矩阵
double W_G[innode][hidenode];     //用于产生新记忆的权值矩阵
double U_G[hidenode][hidenode];   //用于产生新记忆的权值矩阵
double W_out[hidenode];  //连接隐含层与输出层的权值矩阵

// 初始化
for(i=0;i<innode;i++){
	for(j=0;j<hidenode;j++){
    	W_I[i][j] = uniform_plus_minus_one;
    	W_F[i][j] = uniform_plus_minus_one;
    	W_O[i][j] = uniform_plus_minus_one;
    	W_G[i][j] = uniform_plus_minus_one;
	}
}
for(i=0;i<hidenode;i++){
	for(j=0;j<hidenode;j++){
		U_I[i][j] = uniform_plus_minus_one;
		U_F[i][j] = uniform_plus_minus_one;
		U_O[i][j] = uniform_plus_minus_one;
		U_G[i][j] = uniform_plus_minus_one;
	}
	W_out[i] = uniform_plus_minus_one;
}
```

之后，在训练过程中，我们需要定义二维数组来保存各个门和记忆的数组：

```c
double **I_vector = (double **)malloc((cell_num)*sizeof(double *)); //保存输入门信息
double **F_vector = (double **)malloc((cell_num)*sizeof(double *)); //保存遗忘门信息
double **O_vector = (double **)malloc((cell_num)*sizeof(double *)); //保存输出门信息
double **G_vector = (double **)malloc((cell_num)*sizeof(double *)); //保存记忆信息(C`t)    
double **S_vector = (double **)malloc((cell_num+1)*sizeof(double *)); //保存记忆信息(Ct) 
double **h_vector = (double **)malloc((cell_num+1)*sizeof(double *)); //保存细胞输出信息
		
for(j=0;j<cell_num;j++){
	S_vector[j] = (double *)malloc(hidenode*sizeof(double));
	h_vector[j] = (double *)malloc(hidenode*sizeof(double));
	I_vector[j] = (double *)malloc(hidenode*sizeof(double));
	F_vector[j] = (double *)malloc(hidenode*sizeof(double));
	O_vector[j] = (double *)malloc(hidenode*sizeof(double));
	G_vector[j] = (double *)malloc(hidenode*sizeof(double));
}
	S_vector[cell_num] = (double *)malloc(hidenode*sizeof(double));
	h_vector[cell_num] = (double *)malloc(hidenode*sizeof(double));
```

#### 2.3.2 构建细胞

LSTM的细胞主要由以下四个部分组成：

- 遗忘门
- 输入门
- 输出门
- 记忆部分

##### 2.3.2.1 遗忘门

遗忘门主要负责接受并筛选上一个细胞的信息。设$ f$为遗忘门输出值，$h$为细胞输出值，$x$为输入值，$\sigma$表示激活函数，$W_f$和$b_f$代表遗忘门的权重和偏差，其计算公式为：

$$
f_t = \sigma(W_f[h_{t-1},x_t]+b_f)
$$
![img](https://upload-images.jianshu.io/upload_images/6983308-5fb98869b61eced2.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

##### 2.3.2.2 输入门

输入门主要负责控制新信息的传入，并将新信息传入记忆中。设$i$和$\tilde{C}$为输入门输出值，$W_i$、$W_C$和$b_i$、$b_C$代表输入门的权重和偏差，其计算公式为：
$$
i_t=\sigma(W_i[h_{t-1},x_t]+b_i) \\
\tilde{C_t}=tanh(W_C[h_{t-1},x_t]+b_C)
$$
![img](https://upload-images.jianshu.io/upload_images/6983308-43b42ce338d0566d.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

##### 2.3.2.3 记忆部分

记忆主要负责记录该细胞里的信息，并影响后续的细胞。设$C$为记忆值，则公式如下：
$$
C_t=f_t*C_{t-1}+i*\tilde{C_t}
$$
![img](https://upload-images.jianshu.io/upload_images/6983308-cb48d627cc8df11e.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

##### 2.3.2.4 输出门

输出门主要负责输出值，同时输出值也将作为下一个细胞输入的一部分。设$h$为细胞输出值，$W_o$和$b_o$代表输出门的权重和偏差，则公式如下：
$$
o_t=\sigma (W_o[h_{t-1},x_t]+b_o) \\
h_t=o_t*tanh(C_t)
$$
![img](https://upload-images.jianshu.io/upload_images/6983308-977224dfe3f34477.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

##### 2.3.2.5 代码

将以上四个部分结合起来，便可以构成LSTM的细胞。以本文为例，将LSTM前向传播过程写成代码片段如下：

```c
for(p=0;p<cell_num;p++){
	x[0]=a[p]; //输入值
	x[1]=b[p]; //输入值
	double t = (double)c[p];      //实际值
	double in_gate[hidenode];     //输入门
	double out_gate[hidenode];    //输出门
	double forget_gate[hidenode]; //遗忘门
	double g_gate[hidenode];      //C`t 
	double memory[hidenode];       //记忆值
	double h[hidenode];           //隐层输出值
	            
	double *h_pre = h_vector[p];
	double *memory_pre = M_vector[p];
	    		
	for(k=0; k<hidenode; k++)
	{   //输入层转播到隐层
		double inGate = 0.0;
	    double outGate = 0.0;
	    double forgetGate = 0.0;
	    double gGate = 0.0;
	    double s = 0.0;
	                
	    for(m=0; m<innode; m++) 
	    {
	        inGate += x[m] * W_I[m][k]; 
	        outGate += x[m] * W_O[m][k];
	        forgetGate += x[m] * W_F[m][k];
	        gGate += x[m] * W_G[m][k];
	    }
	                
	    for(m=0; m<hidenode; m++)
        {
            inGate += h_pre[m] * U_I[m][k];
            outGate += h_pre[m] * U_O[m][k];
            forgetGate += h_pre[m] * U_F[m][k];
            gGate += h_pre[m] * U_G[m][k];
        }
	 
					
	    in_gate[k] = sigmoid(inGate);
	    out_gate[k] = sigmoid(outGate);
	    forget_gate[k] = sigmoid(forgetGate);
	    g_gate[k] = sigmoid(gGate);
	 
	    double m_pre = memory_pre[k];
	    memory[k] = forget_gate[k] * m_pre + g_gate[k] * in_gate[k];
	                
	    h[k] = out_gate[k] * tanh(memory[k]);
	                
		I_vector[p][k] = in_gate[k];
		F_vector[p][k] = forget_gate[k];
		O_vector[p][k] = out_gate[k];
		M_vector[p+1][k] = memory[k];
		G_vector[p][k] = g_gate[k];
		h_vector[p+1][k] = h[k];
	 }

	 //隐藏层传播到输出层
	 double out = 0.0;
	 for(j=0; j<hidenode; j++){
	    out += h[j] * W_out[j]; 
	 }
             
	 y = sigmoid(out);               //输出层各单元输出
     predict[p] = (int)floor(y + 0.5);   //记录预测值
     printf('prediction[%d] = %f',p,y);
}
```

#### 2.3.3 反向传播

首先，我们先求预测值和实际值的误差：

```c
double y_delta[cell_num];
for(p=0;p<cell_num;p++){
	//前向传播部分略
	//保存标准误差关于输出层的偏导
	y_delta[p] = (t - y) * dsigmoid(y);
}
```

之后进行反向传播，更新权值矩阵：

```c
//误差反向传播
 
//隐含层偏差，通过当前之后一个时间点的隐含层误差和当前输出层的误差计算
double h_delta[hidenode];  
double O_delta[hidenode];
double I_delta[hidenode];
double F_delta[hidenode];
double G_delta[hidenode];
double memory_delta[hidenode];
//当前时间之后的一个隐藏层误差
double O_future_delta[hidenode]; 
double I_future_delta[hidenode];
double F_future_delta[hidenode];
double G_future_delta[hidenode];
double memory_future_delta[hidenode];
double forget_gate_future[hidenode];
for(j=0; j<hidenode; j++)
{
	O_future_delta[j] = 0.0;
	I_future_delta[j] = 0.0;
	F_future_delta[j] = 0.0;
	G_future_delta[j] = 0.0;
	memory_future_delta[j] = 0.0;
	forget_gate_future[j] = 0.0;
}
	        
for(p=cell_num-1; p>=0 ; p--)
{
    x[0] = a[p];
	x[1] = b[p];
	 
	//当前隐藏层
	double in_gate[hidenode];     //输入门
	double out_gate[hidenode];    //输出门
	double forget_gate[hidenode]; //遗忘门
	double g_gate[hidenode];      //C`t
	double memory[hidenode];     //记忆值
	double h[hidenode];         //隐层输出值
	for(k=0;k<hidenode;k++){
		in_gate[k] = I_vector[p][k];
		out_gate[k] = O_vector[p][k];
		forget_gate[k] = F_vector[p][k]; //遗忘门
	    g_gate[k] = G_vector[p][k];      //C`t 
	    memory[k] = M_vector[p+1][k];     //记忆值
	    h[k] = h_vector[p+1][k];         //隐层输出值
	}
	//前一个隐藏层
	double *h_pre = h_vector[p];   
	double *memory_pre = M_vector[p];
	 
	//更新隐含层和输出层之间的连接权
	for(j=0; j<hidenode; j++){
	    W_out[j] += l_rate * y_delta[p] * h[j];  
	}
	                
	//对于网络中每个隐藏单元，计算误差项，并更新权值
	for(j=0; j<hidenode; j++) 
	{
		h_delta[j] = y_delta[p] * W_out[j];
	    for(k=0; k<hidenode; k++)
	    {
	    	h_delta[j] += I_future_delta[k] * U_I[j][k];
	        h_delta[j] += F_future_delta[k] * U_F[j][k];
	        h_delta[j] += O_future_delta[k] * U_O[j][k];
	        h_delta[j] += G_future_delta[k] * U_G[j][k];
	    }
	 
	    O_delta[j] = 0.0;
	    I_delta[j] = 0.0;
	    F_delta[j] = 0.0;
	    G_delta[j] = 0.0;
	    memory_delta[j] = 0.0;
	 
	    //隐含层的校正误差
	    O_delta[j] = h_delta[j] * tanh(memory[j]) * dsigmoid(out_gate[j]);
	    memory_delta[j] = h_delta[j] * out_gate[j] * dtanh(memory[j]) +
	                                 memory_future_delta[j] * forget_gate_future[j];
	                F_delta[j] = memory_delta[j] * memory_pre[j] * dsigmoid(forget_gate[j]);
	    I_delta[j] = memory_delta[j] * g_gate[j] * dsigmoid(in_gate[j]);
	    G_delta[j] = memory_delta[j] * in_gate[j] * dsigmoid(g_gate[j]);
	                
	    O_future_delta[j] = O_delta[j];
	    F_future_delta[j] = F_delta[j];
	    I_future_delta[j] = I_delta[j];
	    G_future_delta[j] = G_delta[j];
	    memory_future_delta[j] = memory_delta[j];
	    forget_gate_future[j] = forget_gate[j];	
	 
	    //更新前一个隐含层和现在隐含层之间的权值
	    for(k=0; k<hidenode; k++)
	    {
	    	U_I[k][j] += l_rate * I_delta[j] * h_pre[k];
	        U_F[k][j] += l_rate * F_delta[j] * h_pre[k];
	        U_O[k][j] += l_rate * O_delta[j] * h_pre[k];
	        U_G[k][j] += l_rate * G_delta[j] * h_pre[k];
	    }
	 
	    //更新输入层和隐含层之间的连接权
	    for(k=0; k<innode; k++)
	    {
	    	W_I[k][j] += l_rate * I_delta[j] * x[k];
	        W_F[k][j] += l_rate * F_delta[j] * x[k];
	        W_O[k][j] += l_rate * O_delta[j] * x[k];
	        W_G[k][j] += l_rate * G_delta[j] * x[k];
	    }
	}
}
```

#### 2.3.4 预测

训练完成后，就可以利用训练好的权重矩阵进行预测。其过程和前向传播大致相同。代码如下：

```c
int *predictions=(int*)malloc(test_size*sizeof(int));
// 预测
for(i=0;i<test_size;i++){
	double **M_vector = (double **)malloc((cell_num+1)*sizeof(double *));     
	double **h_vector = (double **)malloc((cell_num+1)*sizeof(double *));
	for(j=0;j<cell_num+1;j++){
		M_vector[j] = (double *)malloc(hidenode*sizeof(double));
		h_vector[j] = (double *)malloc(hidenode*sizeof(double));
	}
		
	int predict[cell_num];               //保存每次生成的预测值
    memset(predict, 0, sizeof(predict));
        
	double M[hidenode];     //记忆值
    double h[hidenode];     //输出值
        
    for(j=0; j<hidenode; j++)  
    {
    	M[j] = 0;
        h[j] = 0;
        M_vector[0][j] = 0;
        h_vector[0][j] = 0;
    }

        	
    int a_int = test[i][0];
    int a[cell_num];
    int b_int = test[i][1];
    int b[cell_num];
    int c_int = test[i][2];
    int c[cell_num];

    int2binary(a_int, a); //把输入值变成二进制
    int2binary(b_int, b);
    int2binary(c_int, c);
        
        	
   	for(p=0;p<cell_num;p++){
		x[0]=a[p];
	    x[1]=b[p];
	    double in_gate[hidenode];     //输入门
	    double out_gate[hidenode];    //输出门
	    double forget_gate[hidenode]; //遗忘门
	    double g_gate[hidenode];      //C`t 
	    double memory[hidenode];       //记忆值
	    double h[hidenode];           //隐层输出值
	    		
	    for(k=0; k<hidenode; k++)
	    {   
	        //输入层转播到隐层
	    	double inGate = 0.0;
	        double outGate = 0.0;
	        double forgetGate = 0.0;
	        double gGate = 0.0;
	        double s = 0.0;
	 
	        double *h_pre = h_vector[p];
	        double *memory_pre = M_vector[p];
	                
	        for(m=0; m<innode; m++) 
	        {
	        	inGate += x[m] * W_I[m][k]; 
	            outGate += x[m] * W_O[m][k];
	            forgetGate += x[m] * W_F[m][k];
	            gGate += x[m] * W_G[m][k];
	        }
	                
	        for(m=0; m<hidenode; m++)
            {
            	inGate += h_pre[m] * U_I[m][k];
                outGate += h_pre[m] * U_O[m][k];
                forgetGate += h_pre[m] * U_F[m][k];
                gGate += h_pre[m] * U_G[m][k];
            }
	 
	
	        in_gate[k] = sigmoid(inGate);   
	        out_gate[k] = sigmoid(outGate);
	        forget_gate[k] = sigmoid(forgetGate);
	        g_gate[k] = sigmoid(gGate);
	 
	        double m_pre = memory_pre[k];
	        memory[k] = forget_gate[k] * m_pre + g_gate[k] * in_gate[k];
	        h[k] = out_gate[k] * tanh(memory[k]);
	            
	        M_vector[p+1][k] = memory[k];
			h_vector[p+1][k] = h[k];
	    }

	    //隐藏层传播到输出层
	    double out = 0.0;
	    for(j=0; j<hidenode; j++){
	    	out += h[j] * W_out[j];
		}
	    y = sigmoid(out);               //输出层各单元输出
	    predict[p] = (int)floor(y + 0.5);
	}
	free(M_vector);
	free(h_vector);
	    
	double out=0;
	for(k=cell_num-1; k>=0; k--){
    	out += predict[k] * pow(2, k);
	}		
	predictions[i] = out;
}
```

### 2.4  计算RMSE

该步骤代码与前面代码一致，不再重复给出。

### 2.5  按划分的k折交叉验证计算预测所得平均RMSE

```c
#include <stdlib.h>
#include <stdio.h>
extern double  ***cross_validation_split(double **dataset, int row, int n_folds, int fold_size,int col);
extern double* get_test_prediction(double **train, double **test, double l_rate, int n_epoch, int train_size,int test_size,int col);
extern double accuracy_metric(double *actual, double *predicted, int fold_size);

double* evaluate_algorithm(double **dataset, int n_folds, int fold_size, double l_rate, int n_epoch,int col,int row) 
{
	double*** split =  cross_validation_split(dataset, row, n_folds, fold_size,col);
	int i, j, k, l;
	int test_size = fold_size;
	int train_size = fold_size * (n_folds - 1);//train_size个一维数组
	double* score = (double*)malloc(n_folds * sizeof(double));
	for (i = 0; i < n_folds; i++) 
    {  //因为要遍历删除，所以拷贝一份split
		double*** split_copy = (double***)malloc(n_folds * sizeof(double**));
		for (j = 0; j < n_folds; j++) {
			split_copy[j] = (double**)malloc(fold_size * sizeof(double*));
			for (k = 0; k < fold_size; k++) {
				split_copy[j][k] = (double*)malloc(col * sizeof(double));
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
		double** test_set = (double**)malloc(test_size * sizeof(double*));
		for (j = 0; j < test_size; j++) {//对test_size中的每一行
			test_set[j] = (double*)malloc(col * sizeof(double));
			for (k = 0; k < col; k++) {
				test_set[j][k] = split_copy[i][j][k];
			}
		}
		for (j = i; j < n_folds - 1; j++) {
			split_copy[j] = split_copy[j + 1];
		}
		double** train_set = (double**)malloc(train_size * sizeof(double*));
		for (k = 0; k < n_folds - 1; k++) {
			for (l = 0; l < fold_size; l++) {
				train_set[k*fold_size + l] = (double*)malloc(col * sizeof(double));
				train_set[k*fold_size + l] = split_copy[k][l];
			}
		}
		double *predicted_2;
		predicted_2 = get_test_prediction(train_set, test_set, l_rate, n_epoch, train_size,test_size,col);
		double predicted[test_size];
		double* actual = (double*)malloc(test_size * sizeof(double));
		for(l=0;l<test_size;l++){
			predicted[l] = (double) *(predicted_2+l);
			actual[l] = test_set[l][col - 1];
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
	printf("mean_accuracy=%f\n", total / n_folds);
	return score;
}
```

## 3.完整算法及应用

本节将用LSTM网络让模型学习加法，其过程如下：

- 读取数据
- 把数据转换为二进制格式（输入与实际值）
- 训练模型
- 预测

下面给出完整的主函数以及训练函数代码：

main.c:

```c
#include "math.h"
#include "stdlib.h"
#include "time.h"
#include "assert.h"
#include "string.h"
#include "stdio.h" 
 

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);

void main(){
	char filename[] = "data.csv";
    char line[1024];
    int row = get_row(filename);
    int col = get_col(filename);
    printf("row = %d\n",row);
    printf("col = %d\n",col);
    double **dataset = (double **)malloc(row*sizeof(int *));
    int i;
	for (i = 0; i < row; ++i){
		dataset[i] = (double *)malloc(col*sizeof(double));
	}//动态申请二维数组	
	get_two_dimension(line, dataset, filename);
    double l_rate = 0.01;
	int n_epoch = 100;
	int n_folds = 5;
	int fold_size;
    fold_size=(int)(row/n_folds);
	evaluate_algorithm(dataset, n_folds, fold_size, l_rate, n_epoch,col,row);
}
```

test_prediction.c:

```c
#define innode  2       //输入结点数
#define hidenode  12   //隐藏结点数
#define cell_num 8  //LMTM细胞数 

#define uniform_plus_minus_one ( (double)( 2.0 * rand() ) / ((double)RAND_MAX + 1.0) - 1.0 )  //均匀随机分布 

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
 
//tanh的导数，y为tanh值
double dtanh(double y)
{
    y = tanh(y);
    return 1.0 - y * y;  
}
 
//将一个10进制整数转换为2进制数
void int2binary(int n, int *arr)
{
    int i = 0;
    while(n)
    {
        arr[i++] = n % 2;
        n /= 2;
    }
    while(i < cell_num)
        arr[i++] = 0;
}

//训练模型并获得预测值 
int* get_test_prediction(double **train, double **test, double l_rate, int n_epoch, int train_size,int test_size,int col){
	int epoch,i, j, k, m, p;
	int x[innode];
	double y;
	
	double W_I[innode][hidenode];     //连接输入与隐含层单元中输入门的权值矩阵
    double U_I[hidenode][hidenode];   //连接上一隐层输出与本隐含层单元中输入门的权值矩阵
    double W_F[innode][hidenode];     //连接输入与隐含层单元中遗忘门的权值矩阵
    double U_F[hidenode][hidenode];   //连接上一隐含层与本隐含层单元中遗忘门的权值矩阵
    double W_O[innode][hidenode];     //连接输入与隐含层单元中遗忘门的权值矩阵
    double U_O[hidenode][hidenode];   //连接上一隐含层与现在时刻的隐含层的权值矩阵
    double W_G[innode][hidenode];     //用于产生新记忆的权值矩阵
    double U_G[hidenode][hidenode];   //用于产生新记忆的权值矩阵
    double W_out[hidenode];  //连接隐层与输出层的权值矩阵
    
    // 初始化 
    for(i=0;i<innode;i++){
    	for(j=0;j<hidenode;j++){
    		W_I[i][j] = uniform_plus_minus_one;
    		W_F[i][j] = uniform_plus_minus_one;
    		W_O[i][j] = uniform_plus_minus_one;
    		W_G[i][j] = uniform_plus_minus_one;
		}
	}
	for(i=0;i<hidenode;i++){
		for(j=0;j<hidenode;j++){
			U_I[i][j] = uniform_plus_minus_one;
			U_F[i][j] = uniform_plus_minus_one;
			U_O[i][j] = uniform_plus_minus_one;
			U_G[i][j] = uniform_plus_minus_one;
		}
		W_out[i] = uniform_plus_minus_one;
	}
	
    
    for(epoch=0;epoch<n_epoch;epoch++){
    	for(i=0;i<train_size;i++){
    		double **I_vector = (double **)malloc((cell_num)*sizeof(double *));
			double **F_vector = (double **)malloc((cell_num)*sizeof(double *));
			double **O_vector = (double **)malloc((cell_num)*sizeof(double *));    
		    double **G_vector = (double **)malloc((cell_num)*sizeof(double *));      
		    double **M_vector = (double **)malloc((cell_num+1)*sizeof(double *));     
		    double **h_vector = (double **)malloc((cell_num+1)*sizeof(double *));     
		    double y_delta[cell_num];    //保存误差关于输出层的偏导
		
			for(j=0;j<cell_num;j++){
		        M_vector[j] = (double *)malloc(hidenode*sizeof(double));
		        h_vector[j] = (double *)malloc(hidenode*sizeof(double));
		        I_vector[j] = (double *)malloc(hidenode*sizeof(double));
		        F_vector[j] = (double *)malloc(hidenode*sizeof(double));
		        O_vector[j] = (double *)malloc(hidenode*sizeof(double));
		        G_vector[j] = (double *)malloc(hidenode*sizeof(double));
			}
			M_vector[cell_num] = (double *)malloc(hidenode*sizeof(double));
		    h_vector[cell_num] = (double *)malloc(hidenode*sizeof(double));
 
        	int predict[cell_num];     //保存每次生成的预测值
        	memset(predict, 0, sizeof(predict));
 
        
        	double M[hidenode];     //记忆值
        	double h[hidenode];     //输出值
        
        	for(j=0; j<hidenode; j++)  
        	{
            	M[j] = 0;
            	h[j] = 0;
            	M_vector[0][j] = 0;
        		h_vector[0][j] = 0;
        	}
        	
        	int a_int = train[i][0];
        	int a[cell_num];
        	int b_int = train[i][1];
        	int b[cell_num];
        	
        	int c_int = train[i][2];
        	
        	int c[cell_num];
        	int2binary(a_int, a);
        	int2binary(b_int, b);
        	int2binary(c_int, c);
        	
        	for(p=0;p<cell_num;p++){
	    		x[0]=a[p];
	    		x[1]=b[p];
	    		double t = (double)c[p];      //实际值
	    		double in_gate[hidenode];     //输入门
	            double out_gate[hidenode];    //输出门
	            double forget_gate[hidenode]; //遗忘门
	            double g_gate[hidenode];      //C`t 
	            double memory[hidenode];       //记忆值
	            double h[hidenode];           //隐层输出值
	            
	            double *h_pre = h_vector[p];
	            double *memory_pre = M_vector[p];
	    		
	    		for(k=0; k<hidenode; k++)
	            {   
	                //输入层转播到隐层
	                double inGate = 0.0;
	                double outGate = 0.0;
	                double forgetGate = 0.0;
	                double gGate = 0.0;
	                double s = 0.0;
	                

	 
	                for(m=0; m<innode; m++) 
	                {
	                    inGate += x[m] * W_I[m][k]; 
	                    outGate += x[m] * W_O[m][k];
	                    forgetGate += x[m] * W_F[m][k];
	                    gGate += x[m] * W_G[m][k];
	                }
	                
	                for(m=0; m<hidenode; m++)
                	{
                    	inGate += h_pre[m] * U_I[m][k];
                    	outGate += h_pre[m] * U_O[m][k];
                    	forgetGate += h_pre[m] * U_F[m][k];
                    	gGate += h_pre[m] * U_G[m][k];
                	}
	 
					
	                in_gate[k] = sigmoid(inGate);
	                out_gate[k] = sigmoid(outGate);
	                forget_gate[k] = sigmoid(forgetGate);
	                g_gate[k] = sigmoid(gGate);
	 
	                double m_pre = memory_pre[k];
	                memory[k] = forget_gate[k] * m_pre + g_gate[k] * in_gate[k];
	                
	                h[k] = out_gate[k] * tanh(memory[k]);
	                
					I_vector[p][k] = in_gate[k];
					F_vector[p][k] = forget_gate[k];
					O_vector[p][k] = out_gate[k];
					M_vector[p+1][k] = memory[k];
					G_vector[p][k] = g_gate[k];
					h_vector[p+1][k] = h[k];
	            }

	            //隐藏层传播到输出层
	            double out = 0.0;
	            for(j=0; j<hidenode; j++){
	                out += h[j] * W_out[j]; 
				}
             
	            y = sigmoid(out);               //输出层各单元输出
	            predict[p] = (int)floor(y + 0.5);   //记录预测值
	            
				//保存标准误差关于输出层的偏导
	            y_delta[p] = (t - y) * dsigmoid(y);
			}
			//误差反向传播
 
	        //隐含层偏差，通过当前之后一个时间点的隐含层误差和当前输出层的误差计算
	        double h_delta[hidenode];  
	        double O_delta[hidenode];
	        double I_delta[hidenode];
	        double F_delta[hidenode];
	        double G_delta[hidenode];
	        double memory_delta[hidenode];
	        //当前时间之后的一个隐藏层误差
	        double O_future_delta[hidenode]; 
	        double I_future_delta[hidenode];
	        double F_future_delta[hidenode];
	        double G_future_delta[hidenode];
	        double memory_future_delta[hidenode];
	        double forget_gate_future[hidenode];
	        for(j=0; j<hidenode; j++)
	        {
	            O_future_delta[j] = 0.0;
	            I_future_delta[j] = 0.0;
	            F_future_delta[j] = 0.0;
	            G_future_delta[j] = 0.0;
	            memory_future_delta[j] = 0.0;
	            forget_gate_future[j] = 0.0;
	        }
	        
	        for(p=cell_num-1; p>=0 ; p--)
	        {
	            x[0] = a[p];
	            x[1] = b[p];
	 
	            //当前隐藏层

				double in_gate[hidenode];     //输入门
	            double out_gate[hidenode];    //输出门
	            double forget_gate[hidenode]; //遗忘门
	            double g_gate[hidenode];      //C`t
	            double memory[hidenode];     //记忆值
	            double h[hidenode];         //隐层输出值
				for(k=0;k<hidenode;k++){
					in_gate[k] = I_vector[p][k];
					out_gate[k] = O_vector[p][k];
					forget_gate[k] = F_vector[p][k]; //遗忘门
	            	g_gate[k] = G_vector[p][k];      //C`t 
	            	memory[k] = M_vector[p+1][k];     //记忆值
	            	h[k] = h_vector[p+1][k];         //隐层输出值
				}
	            //前一个隐藏层
	            double *h_pre = h_vector[p];   
	            double *memory_pre = M_vector[p];
	 
	            //更新隐含层和输出层之间的连接权
	            for(j=0; j<hidenode; j++){
	            	W_out[j] += l_rate * y_delta[p] * h[j];  
				}
	                
	            //对于网络中每个隐藏单元，计算误差项，并更新权值
	            for(j=0; j<hidenode; j++) 
	            {
	                h_delta[j] = y_delta[p] * W_out[j];
	                for(k=0; k<hidenode; k++)
	                {
	                    h_delta[j] += I_future_delta[k] * U_I[j][k];
	                    h_delta[j] += F_future_delta[k] * U_F[j][k];
	                    h_delta[j] += O_future_delta[k] * U_O[j][k];
	                    h_delta[j] += G_future_delta[k] * U_G[j][k];
	                }
	 
	                O_delta[j] = 0.0;
	                I_delta[j] = 0.0;
	                F_delta[j] = 0.0;
	                G_delta[j] = 0.0;
	                memory_delta[j] = 0.0;
	 
	                //隐含层的校正误差
	                O_delta[j] = h_delta[j] * tanh(memory[j]) * dsigmoid(out_gate[j]);
	                memory_delta[j] = h_delta[j] * out_gate[j] * dtanh(memory[j]) +
	                                 memory_future_delta[j] * forget_gate_future[j];
	                F_delta[j] = memory_delta[j] * memory_pre[j] * dsigmoid(forget_gate[j]);
	                I_delta[j] = memory_delta[j] * g_gate[j] * dsigmoid(in_gate[j]);
	                G_delta[j] = memory_delta[j] * in_gate[j] * dsigmoid(g_gate[j]);
	                
	                O_future_delta[j] = O_delta[j];
	            	F_future_delta[j] = F_delta[j];
	            	I_future_delta[j] = I_delta[j];
	            	G_future_delta[j] = G_delta[j];
	            	memory_future_delta[j] = memory_delta[j];
	            	forget_gate_future[j] = forget_gate[j];	
	 
	                //更新前一个隐含层和现在隐含层之间的权值
	                for(k=0; k<hidenode; k++)
	                {
	                	U_I[k][j] += l_rate * I_delta[j] * h_pre[k];
	                    U_F[k][j] += l_rate * F_delta[j] * h_pre[k];
	                    U_O[k][j] += l_rate * O_delta[j] * h_pre[k];
	                    U_G[k][j] += l_rate * G_delta[j] * h_pre[k];
	                }
	 
	                //更新输入层和隐含层之间的连接权
	                for(k=0; k<innode; k++)
	                {
	                    W_I[k][j] += l_rate * I_delta[j] * x[k];
	                    W_F[k][j] += l_rate * F_delta[j] * x[k];
	                    W_O[k][j] += l_rate * O_delta[j] * x[k];
	                    W_G[k][j] += l_rate * G_delta[j] * x[k];
	                }
	 
	            }
	        }
			free(I_vector);
	        free(F_vector);
	        free(O_vector);
	        free(G_vector);
	        free(M_vector);
	        free(h_vector);   
		}
	}
	int *predictions=(int*)malloc(test_size*sizeof(int));
	// 预测
	for(i=0;i<test_size;i++){
		double **M_vector = (double **)malloc((cell_num+1)*sizeof(double *));     
		double **h_vector = (double **)malloc((cell_num+1)*sizeof(double *));
		for(j=0;j<cell_num+1;j++){
		    M_vector[j] = (double *)malloc(hidenode*sizeof(double));
		    h_vector[j] = (double *)malloc(hidenode*sizeof(double));
		}
		
		
		int predict[cell_num];               //保存每次生成的预测值
        memset(predict, 0, sizeof(predict));
        
		double M[hidenode];     //记忆值
        double h[hidenode];     //输出值
        
        for(j=0; j<hidenode; j++)  
        {
            M[j] = 0;
            h[j] = 0;
            M_vector[0][j] = 0;
        	h_vector[0][j] = 0;
        }

        	
        int a_int = test[i][0];
        int a[cell_num];
        int b_int = test[i][1];
        int b[cell_num];
        int c_int = test[i][2];
        int c[cell_num];

        int2binary(a_int, a);
        int2binary(b_int, b);
        int2binary(c_int, c);
        
        	
        for(p=0;p<cell_num;p++){
	    	x[0]=a[p];
	    	x[1]=b[p];
	    	double in_gate[hidenode];     //输入门
	        double out_gate[hidenode];    //输出门
	        double forget_gate[hidenode]; //遗忘门
	        double g_gate[hidenode];      //C`t 
	        double memory[hidenode];       //记忆值
	        double h[hidenode];           //隐层输出值
	    		
	    	for(k=0; k<hidenode; k++)
	        {   
	            //输入层转播到隐层
	            double inGate = 0.0;
	            double outGate = 0.0;
	            double forgetGate = 0.0;
	            double gGate = 0.0;
	            double s = 0.0;
	 
	 
	            double *h_pre = h_vector[p];
	            double *memory_pre = M_vector[p];
	                
	            for(m=0; m<innode; m++) 
	            {
	                inGate += x[m] * W_I[m][k]; 
	                outGate += x[m] * W_O[m][k];
	                forgetGate += x[m] * W_F[m][k];
	                gGate += x[m] * W_G[m][k];
	            }
	                
	            for(m=0; m<hidenode; m++)
                {
                    inGate += h_pre[m] * U_I[m][k];
                    outGate += h_pre[m] * U_O[m][k];
                    forgetGate += h_pre[m] * U_F[m][k];
                    gGate += h_pre[m] * U_G[m][k];
                }
	 
	
	            in_gate[k] = sigmoid(inGate);   
	            out_gate[k] = sigmoid(outGate);
	            forget_gate[k] = sigmoid(forgetGate);
	            g_gate[k] = sigmoid(gGate);
	 
	            double m_pre = memory_pre[k];
	            memory[k] = forget_gate[k] * m_pre + g_gate[k] * in_gate[k];
	            h[k] = out_gate[k] * tanh(memory[k]);
	            
	            M_vector[p+1][k] = memory[k];
				h_vector[p+1][k] = h[k];
	        }

	        //隐藏层传播到输出层
	        double out = 0.0;
	        for(j=0; j<hidenode; j++){
	            out += h[j] * W_out[j];
			}
	        y = sigmoid(out);               //输出层各单元输出
	        predict[p] = (int)floor(y + 0.5);
	    }
	    free(M_vector);
	    free(h_vector);
	    
	    double out=0;
		for(k=cell_num-1; k>=0; k--){
            out += predict[k] * pow(2, k);
		}		
		predictions[i] = out;
	}
	return predictions;
}
```