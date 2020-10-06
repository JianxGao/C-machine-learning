#include<stdlib.h>
#include<stdio.h>

extern void coefficients(float** data, float* coef, int length);
float* get_test_prediction(int col,int row,float** train, float** test, int n_folds) {
	float* coef = (float*)malloc(col * sizeof(float));
	int i;
	for (i = 0; i < col; i++) {
		coef[i] = 0.0;
	}
	int fold_size = (int)row / n_folds;
	int train_size = fold_size * (n_folds - 1);
	coefficients(train, coef, train_size);//核心算法部分
	float* predictions = (float*)malloc(fold_size * sizeof(float));//因为test_size和fold_size一样大
	for (i = 0; i < fold_size; i++) {//因为test_size和fold_size一样大
		predictions[i] = coef[0] + coef[1] * test[i][0];
	}
	return predictions;//返回对test的预测数组
}