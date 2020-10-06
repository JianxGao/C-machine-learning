#include<stdlib.h>
#include<stdio.h>
#include<math.h>

double rmse_metric(double *actual, double *predicted, int fold_size)
{
	double sum_err = 0.0;
	int i;
	for (i = 0; i < fold_size; i++)
	{
		double err = predicted[i] - actual[i];
		//printf("predicted=%f,actual=%f\n", predicted[i], actual[i]);
		sum_err += err * err;
	}
	double mean_err = sum_err / fold_size;
	return sqrt(mean_err);
}