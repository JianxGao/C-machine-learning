#include <stdlib.h>
#include <stdio.h>

double perceptron_single_predict(int col, double *array, double *weights)
{
	double activation = weights[0];
	for (int i = 0; i < col - 1; i++)
	{
		activation += weights[i + 1] * array[i];
	}
	double output = 0.0;
	if (activation >= 0.0)
	{
		output = 1.0;
	}
	else
	{
		output = 0.0;
	}
	return output;
}

void train_weights(double **data, int col, double *weights, double l_rate, int n_epoch, int train_size)
{
	for (int i = 0; i < n_epoch; i++)
	{
		for (int j = 0; j < train_size; j++)
		{
			double yhat = perceptron_single_predict(col, data[j], weights);
			double err = data[j][col - 1] - yhat;
			weights[0] += l_rate * err;
			for (int k = 0; k < col - 1; k++)
			{
				weights[k + 1] += l_rate * err * data[j][k];
			}
		}
	}
}

double *perceptron_predict(double **train, int train_size, double **test, int test_size, int col, double l_rate, int n_epoch)
{
	double *weights = (double *)malloc(col * sizeof(double));
	int i;
	for (i = 0; i < col; i++)
	{
		weights[i] = 0.0;
	}
	train_weights(train, col, weights, l_rate, n_epoch, train_size);
	double *predictions = (double *)malloc(test_size * sizeof(double));
	for (i = 0; i < test_size; i++)
	{
		predictions[i] = perceptron_single_predict(col, test[i], weights);
	}
	return predictions;
}