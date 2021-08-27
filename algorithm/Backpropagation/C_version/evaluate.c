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
		printf("score[%d] = %f%%\n", i, score[i]);
		free(split_copy);
	}
	double total = 0.0;
	for (l = 0; l < n_folds; l++)
	{
		total += score[l];
	}
	printf("mean_accuracy = %f%%\n", total / n_folds);
	return score;
}