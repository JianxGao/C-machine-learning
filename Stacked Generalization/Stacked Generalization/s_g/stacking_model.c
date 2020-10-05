#include<stdlib.h>
#include<stdio.h>
#include<math.h>

double stack_predict(int col, double *array, double *coefficients)
{
	double yhat = coefficients[0];
	int i;
	for (i = 0; i < col - 1; i++){
		yhat += coefficients[i + 1] * array[i];
	}
	return 1 / (1 + exp(-yhat));
}

void coefficients_sgd(double ** dataset, int col, double *coef, double l_rate, int n_epoch, int train_size) 
{
	for (int i = 0; i < n_epoch; i++)
	{
		for (int j = 0; j < train_size; j++)
		{
			double yhat = stack_predict(col, dataset[j], coef);
			double err = dataset[j][col - 1] - yhat;
			coef[0] += l_rate * err * yhat * (1 - yhat);
			
			for (int k = 0; k < col - 1; k++)
			{
				coef[k + 1] += l_rate * err * yhat * (1 - yhat) * dataset[j][k];
				//printf("%f\t%f\t%f\t%f\n", l_rate,err,yhat, dataset[j][k]);
			}
		}
	}
}
