#include<stdlib.h>
#include<stdio.h>

extern void train_weights(double **data, int col, double *weights, double l_rate, int n_epoch, int train_size);
extern double predict(int col, double *array, double *weights);
double* get_test_prediction(double** train, double** test,int row,int col, double l_rate, int n_epoch, int n_folds) {
	double* weights = (double*)malloc(col * sizeof(double));
	int i;
	for (i = 0; i < col; i++) {
		weights[i] = 0.0;
	}
	int fold_size = (int)row / n_folds;
	int train_size = fold_size * (n_folds - 1);
	//coefficients_sgd(train, coef, l_rate, n_epoch, train_size);//核心算法执行部分
	train_weights(train, col,weights, l_rate, n_epoch, train_size);//核心算法执行部分
	double* predictions = (double*)malloc(fold_size * sizeof(double));//因为test_size和fold_size一样大
	for (i = 0; i < fold_size; i++) {//因为test_size和fold_size一样大
		predictions[i] = predict(col,test[i], weights);
	}
	return predictions;//返回对test的预测数组
}