#include<stdlib.h>
#include<stdio.h>
#include<math.h>

extern double* knn_predict(double **train, int train_size, double **test, int test_size, int col, int num_neighbors);
extern double* perceptron_predict(double** train, int train_size, double** test, int test_size, int col, double l_rate, int n_epoch);
extern void coefficients_sgd(double ** dataset, int col, double *coef, double l_rate, int n_epoch, int train_size);
extern double stack_predict(int col, double *array, double *coefficients);

double* get_test_prediction(double **train, int train_size, double **test, int test_size, int col, int num_neighbors, double l_rate, int n_epoch)
{
	double* coef = (double*)malloc(3 * sizeof(double));
	for (int i = 0; i < 3; i++)
	{
		coef[i] = 0.0;
	}
	// 训练
	double* train_knn_predictions = knn_predict(train, train_size, train, train_size, col, num_neighbors);
	double* train_perceptron_predictions = perceptron_predict(train, train_size, train, train_size, col, l_rate, n_epoch);
	double** new_train = (double **)malloc(train_size * sizeof(double *));
	for (int i = 0; i < train_size; i++)
	{
		new_train[i] = (double *)malloc(3 * sizeof(double));
	}
	// 生成新的数据集
	for (int i = 0; i < train_size; i++)
	{
		new_train[i][0] = train_knn_predictions[i];
		new_train[i][1] = train_perceptron_predictions[i];
		new_train[i][2] = train[i][col - 1];
	}

	coefficients_sgd(new_train, 3, coef, l_rate, n_epoch, train_size);

	// 预测
	double* knn_predictions = knn_predict(train, train_size, test, test_size, col, num_neighbors);
	double* perceptron_predictions = perceptron_predict(train, train_size, test, test_size, col, l_rate, n_epoch);
	
	double** new_test = (double **)malloc(test_size * sizeof(double *));
	for (int i = 0; i < test_size; ++i)
	{
		new_test[i] = (double *)malloc(2 * sizeof(double));
	}
	// 生成新的数据集
	for (int i = 0; i < test_size; i++)
	{
		new_test[i][0] = knn_predictions[i];
		new_test[i][1] = perceptron_predictions[i];
	}
	double* predictions = (double*)malloc(test_size * sizeof(double));//因为test_size和fold_size一样大
	for (int i = 0; i < test_size; i++)
	{
		predictions[i] = round(stack_predict(3, new_test[i], coef));
	}

	return predictions;//返回对test的预测数组
}