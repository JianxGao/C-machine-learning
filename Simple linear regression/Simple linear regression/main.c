#include<stdlib.h>
#include<stdio.h>

double **dataset;
int row, col;

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);

extern float* evaluate_algorithm(double **dataset, int row, int col, int n_folds);
float mean(float* values, int length);
float covariance(float* x, float mean_x, float* y, float mean_y, int length);
float variance(float* values, float mean, int length);
void coefficients(float** data, float* coef, int length);


int main() {	
	char filename[] = "Auto insurance.csv";
	char line[1024];
	row = get_row(filename);
	col = get_col(filename);
	dataset = (double **)malloc(row * sizeof(double *));
	for (int i = 0; i < row; ++i) {
		dataset[i] = (double *)malloc(col * sizeof(double));
	}//动态申请二维数组	
	get_two_dimension(line, dataset, filename);
	int n_folds = 10;
	int fold_size = (int)row / n_folds;
	evaluate_algorithm(dataset, row, col, n_folds);
	return 0;
}
//计算均值方差等统计量（多个函数）
float mean(float* values, int length) {//对一维数组求均值
	int i;
	float sum = 0.0;
	for (i = 0; i < length; i++) {
		sum += values[i];
	}
	float mean = (float)(sum / length);
	return mean;
}
float covariance(float* x, float mean_x, float* y, float mean_y, int length) {
	float cov = 0.0;
	int i = 0;
	for (i = 0; i < length; i++) {
		cov += (x[i] - mean_x)*(y[i] - mean_y);
	}
	return cov;
}
float variance(float* values, float mean, int length) {//这里求的是平方和，没有除以n
	float sum = 0.0;
	int i;
	for (i = 0; i < length; i++) {
		sum += (values[i] - mean)*(values[i] - mean);
	}
	return sum;
}
//由均值方差估计回归系数
void coefficients(float** data, float* coef, int length) {
	float* x = (float*)malloc(length * sizeof(float));
	float* y = (float*)malloc(length * sizeof(float));
	int i;
	for (i = 0; i < length; i++) {
		x[i] = data[i][0];
		y[i] = data[i][1];
		//printf("x[%d]=%f,y[%d]=%f\n",i, x[i],i,y[i]);
	}
	float x_mean = mean(x, length);
	float y_mean = mean(y, length);
	//printf("x_mean=%f,y_mean=%f\n", x_mean, y_mean);
	coef[1] = covariance(x, x_mean, y, y_mean, length) / variance(x, x_mean, length);
	coef[0] = y_mean - coef[1] * x_mean;
	/*for (i = 0; i < 2; i++) {
		printf("coef[%d]=%f\n", i, coef[i]);
	}*/
}

