#include <stdlib.h>
#include <stdio.h>

extern void coefficients_sgd(double **dataset, int col, double *coef, double l_rate, int n_epoch, int train_size);
extern double predict(int col, double array[], double coefficients[]);

double *get_test_prediction(double **dataset, int row, int col, double **train, double **test, double l_rate, int n_epoch, int n_folds)
{
	double *coef = (double *)malloc(col * sizeof(double));
	int i;
	for (i = 0; i < col; i++)
	{
		coef[i] = 0.0;
	}
	int fold_size = (int)row / n_folds;
	int train_size = fold_size * (n_folds - 1);
	coefficients_sgd(train, col, coef, l_rate, n_epoch, train_size);
	double *predictions = (double *)malloc(fold_size * sizeof(double)); 
	for (i = 0; i < fold_size; i++)
	{
		predictions[i] = predict(col, test[i], coef);
		predictions[i] = (float)(int)(predictions[i] + 0.5);
	}
	return predictions;
}