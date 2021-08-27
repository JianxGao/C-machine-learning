#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double accuracy_metric(double *actual, double *predicted, int fold_size)
{
	int correct = 0;
	int i;
	for (i = 0; i < fold_size; i++)
	{
		if (actual[i] == predicted[i])
			correct += 1;
	}
	return (correct / (double)fold_size) * 100.0;
}