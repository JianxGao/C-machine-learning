#include "DT.h"

float accuracy_metric(double *actual, double *predicted, int fold_size)
{
	int correct = 0;
	for (int i = 0; i < fold_size; i++)
	{
		if (actual[i] == predicted[i])
			correct += 1;
	}
	return (correct / (float)fold_size) * 100.0;
}