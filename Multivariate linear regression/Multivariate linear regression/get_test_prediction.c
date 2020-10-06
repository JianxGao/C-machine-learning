#include<stdlib.h>
#include<stdio.h>

extern double* coefficients_sgd(double** dataset, int col, double coef[], double l_rate, int n_epoch, int train_size);
extern double predict(int col, double array[], double coefficients[]);
double* get_test_prediction(double** dataset,int row, int col,double** train, double** test, double l_rate, int n_epoch, int n_folds) {
	double* coef = (double*)malloc(col * sizeof(double));
	int i;
	for (i = 0; i < col; i++) {
		coef[i] = 0.0;
	}
	int fold_size = (int)row / n_folds;
	int train_size = fold_size * (n_folds - 1);
	coefficients_sgd(train, col, coef, l_rate, n_epoch, train_size);//核心算法执行部分
	double* predictions = (double*)malloc(fold_size * sizeof(double));//因为test_size和fold_size一样大
	for (i = 0; i < fold_size; i++) {//因为test_size和fold_size一样大
		predictions[i] = predict(col, test[i],coef);
	}
	return predictions;//返回对test的预测数组
}