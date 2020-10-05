#include<stdlib.h>
#include<stdio.h>

extern double predict(double ***summaries, double *test_row, int class_num, int *class_num_list, int row, int col);
extern int get_class_num(double **dataset, int row, int col);
extern int* get_class_num_list(double **dataset, int class_num, int row, int col);
extern double*** summarize_by_class(double **train, int class_num, int *class_num_list, int row, int col);

double* get_test_prediction(double **train, int train_size, double **test, int test_size, int col)
{
	int class_num = get_class_num(train, train_size, col);
	int *class_num_list = get_class_num_list(train, class_num, train_size, col);
	double* predictions = (double*)malloc(test_size * sizeof(double));//因为test_size和fold_size一样大
	double ***summaries = summarize_by_class(train, class_num, class_num_list, train_size, col);
	for (int i = 0; i < test_size; i++)
	{
		predictions[i] = predict(summaries, test[i], class_num, class_num_list, train_size, col);
	}
	return predictions;//返回对test的预测数组
}