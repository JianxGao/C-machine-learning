#include <stdlib.h>
#include <stdio.h>

extern void coefficients(double **data, double *coef, int length);
double *get_test_prediction(int col, int row, double **train, double **test, int n_folds)
{
	double *coef = (double *)malloc(col * sizeof(double));
	int i;
	for (i = 0; i < col; i++)
	{
		coef[i] = 0.0;
	}
	int fold_size = (int)row / n_folds;
	int train_size = fold_size * (n_folds - 1);
	coefficients(train, coef, train_size);
	double *predictions = (double *)malloc(fold_size * sizeof(double));
	for (i = 0; i < fold_size; i++)
	{
		predictions[i] = coef[0] + coef[1] * test[i][0];
	}
	return predictions;
}