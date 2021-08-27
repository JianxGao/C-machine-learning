#include <stdlib.h>
#include <stdio.h>

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern double *evaluate_algorithm(double **dataset, int row, int col, int n_folds, int n_epoch, double l_rate);
extern void normalize_dataset(double **dataset, int row, int col);

double predict(int col, double array[], double coefficients[])
{
	double yhat = coefficients[0];
	int i;
	for (i = 0; i < col - 1; i++)
		yhat += coefficients[i + 1] * array[i];
	return yhat;
}

double *coefficients_sgd(double **dataset, int col, double coef[], double l_rate, int n_epoch, int train_size)
{
	int i;
	for (i = 0; i < n_epoch; i++)
	{
		int j = 0;
		for (j = 0; j < train_size; j++)
		{
			double yhat = predict(col, dataset[j], coef);
			double err = yhat - dataset[j][col - 1];
			coef[0] -= l_rate * err;
			int k;
			for (k = 0; k < col - 1; k++)
			{
				coef[k + 1] -= l_rate * err * dataset[j][k];
			}
		}
	}
	return coef;
}

int main()
{
	char filename[] = "winequality-white.csv";
	char line[1024];
	double **dataset;
	int row, col;
	row = get_row(filename);
	col = get_col(filename);
	dataset = (double **)malloc(row * sizeof(double *));
	for (int i = 0; i < row; ++i)
	{
		dataset[i] = (double *)malloc(col * sizeof(double));
	}
	get_two_dimension(line, dataset, filename);
	normalize_dataset(dataset, row, col);

	int n_folds = 10;
	double l_rate = 0.001f;
	int n_epoch = 50;
	evaluate_algorithm(dataset, row, col, n_folds, n_epoch, l_rate);
	return 0;
}

