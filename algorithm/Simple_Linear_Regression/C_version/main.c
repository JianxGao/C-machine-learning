#include <stdlib.h>
#include <stdio.h>

double **dataset;
int row, col;

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);

extern double *evaluate_algorithm(double **dataset, int row, int col, int n_folds);
double mean(double *values, int length);
double covariance(double *x, double mean_x, double *y, double mean_y, int length);
double variance(double *values, double mean, int length);
void coefficients(double **data, double *coef, int length);

//计算均值方差等统计量（多个函数）
double mean(double *values, int length)
{ 
    //对一维数组求均值
	int i;
	double sum = 0.0;
	for (i = 0; i < length; i++)
	{
		sum += values[i];
	}
	double mean = (double)(sum / length);
	return mean;
}

double covariance(double *x, double mean_x, double *y, double mean_y, int length)
{
	double cov = 0.0;
	int i = 0;
	for (i = 0; i < length; i++)
	{
		cov += (x[i] - mean_x) * (y[i] - mean_y);
	}
	return cov;
}

double variance(double *values, double mean, int length)
{ 
    //这里求的是平方和，没有除以n
	double sum = 0.0;
	int i;
	for (i = 0; i < length; i++)
	{
		sum += (values[i] - mean) * (values[i] - mean);
	}
	return sum;
}

//由均值方差估计回归系数
void coefficients(double **data, double *coef, int length)
{
	double *x = (double *)malloc(length * sizeof(double));
	double *y = (double *)malloc(length * sizeof(double));
	int i;
	for (i = 0; i < length; i++)
	{
		x[i] = data[i][0];
		y[i] = data[i][1];
	}
	double x_mean = mean(x, length);
	double y_mean = mean(y, length);
	coef[1] = covariance(x, x_mean, y, y_mean, length) / variance(x, x_mean, length);
	coef[0] = y_mean - coef[1] * x_mean;
}

int main()
{
	char filename[] = "insurance.csv";
	char line[1024];
	row = get_row(filename);
	col = get_col(filename);
	dataset = (double **)malloc(row * sizeof(double *));
	for (int i = 0; i < row; ++i)
	{
		dataset[i] = (double *)malloc(col * sizeof(double));
	} 
	//动态申请二维数组
	get_two_dimension(line, dataset, filename);
	int n_folds = 10;
	int fold_size = (int)row / n_folds;
	evaluate_algorithm(dataset, row, col, n_folds);
	return 0;
}