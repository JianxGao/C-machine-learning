#define randval(high) ( (double)rand() / RAND_MAX * high )
#define uniform_plus_minus_one ( (double)( 2.0 * rand() ) / ((double)RAND_MAX + 1.0) - 1.0 )  //均匀随机分布 

#define node1 12 //第一层节点数 
#define node2 3 //第二层节点数 
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
 

double *transfer_to_one_hot(int y){
	double *one_hot = (double *)malloc(class_num*sizeof(double));
	int i;
	for(i=0;i<class_num;i++){
		one_hot[i] = 0;
	}
	one_hot[y-1] = 1;
	return one_hot;
}

//训练模型并获得预测值 
double* get_test_prediction(double **train, double **test, double l_rate, int n_epoch, int train_size,int test_size,int col){
	int epoch,i,j,k;
	// 初始化权重
	double W_1[node1][col]; //第一层权重
	double W_2[node2][node1+1]; //第二层权重 
	double out;
	double predict;
	double layer1_out[node1]; //第一层节点输出值
	double layer2_out[node2]; //第二层节点输出值 
	
	// 初始化权重 
	for(j=0;j<node1;j++){
		for(k=0;k<col;k++){
			W_1[j][k] = uniform_plus_minus_one;
		}
	}
	for(j=0;j<node2;j++){
		for(k=0;k<node1+1;k++){
			W_2[j][k] = uniform_plus_minus_one;
		}
	}
	
	for(epoch=0;epoch<n_epoch;epoch++){
		for(i=0;i<train_size;i++){
			// 前向传播
			for(j=0;j<node1;j++){
				double sum = W_1[j][col-1];
				for(k=0;k<col-1;k++){
					sum += W_1[j][k]*train[i][k];
				}
				layer1_out[j] = sigmoid(sum);
			}
			
			for(j=0;j<node2;j++){
				double sum = W_2[j][node1];
				for(k=0;k<node1;k++){
					sum += W_2[j][k]*layer1_out[k];
				}
				layer2_out[j] = sigmoid(sum);
			}
			// 误差反向传播
			int y;
			y = (int)train[i][col-1];
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
			
			// 更新权重
			for(j=0;j<node1;j++){
				for(k=0;k<col-1;k++){
					W_1[j][k] += l_rate*layer1_delta[j]*train[i][k];
				}
				W_1[j][col] += l_rate*layer1_delta[j];
			}
			for(j=0;j<node2;j++){
				for(k=0;k<node1+1;k++){
					W_2[j][k] += l_rate*layer2_delta[j]*layer1_out[k];
				}
				W_2[j][node1] += l_rate*layer2_delta[j];
			}	
		}
	}
	
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
	return predictions;
}
