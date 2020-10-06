#include<stdlib.h>
#include<stdio.h>
#include<math.h>

float rmse_metric(float *actual, float *predicted, int fold_size)
{
	float sum_err = 0.0;
	int i;
	for (i = 0; i < fold_size; i++)
	{
		float err = predicted[i] - actual[i];
		sum_err += err * err;
	}
	float mean_err = sum_err / fold_size;
	return sqrt(mean_err);
}