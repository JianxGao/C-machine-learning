#include<stdlib.h>
#include<stdio.h>
#include<math.h>

extern void QuickSort(double **arr, int L, int R);
extern double euclidean_distance(double *row1, double *row2, int col);
extern double* get_neighbors(double **train_data, int train_row, int col, double *test_row, int num_neighbors);
extern double predict(double **train_data, int train_row, int col, double *test_row, int num_neighbors);

double* get_test_prediction(double **train, int train_size,  double **test, int test_size, int col, int num_neighbors)
{
	double* predictions = (double*)malloc(test_size * sizeof(double));
	for (int i = 0; i < test_size; i++)
	{
        predictions[i] = predict(train, train_size,col,test[i],num_neighbors);
	}
	return predictions;//返回对test的预测数组
}