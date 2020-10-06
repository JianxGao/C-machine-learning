#include<stdlib.h>
#include<stdio.h>

double **dataset;
int row, col;

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern double* evaluate_algorithm(double **dataset, int row, int col, int n_folds, int n_epoch, double l_rate);
extern void normalize_dataset(double **dataset, int row, int col);

void train_weights(double **data, int col, double *weights, double l_rate, int n_epoch, int train_size);
double predict(int col, double *array, double *weights);
int main() {
	char filename[] = "sonar.csv";
	char line[1024];
	row = get_row(filename);
	col = get_col(filename);
	dataset = (double **)malloc(row * sizeof(double *));
	for (int i = 0; i < row; ++i) {
		dataset[i] = (double *)malloc(col * sizeof(double));
	}//动态申请二维数组	
	get_two_dimension(line, dataset, filename);
	normalize_dataset(dataset, row, col);

	int n_folds = 3;
	double l_rate = 0.01f;
	int n_epoch = 500;
	evaluate_algorithm(dataset, row, col, n_folds, n_epoch, l_rate);
	return 0;
}


void train_weights(double **data, int col,double *weights, double l_rate, int n_epoch, int train_size) {
	int i;
	for (i = 0; i < n_epoch; i++) {
		int j = 0;//遍历每一行
		for (j = 0; j < train_size; j++) {
			double yhat = predict(col,data[j], weights);
			double err = data[j][col - 1] - yhat;
			weights[0] += l_rate * err;
			int k;
			for (k = 0; k < col - 1; k++) {
				weights[k + 1] += l_rate * err * data[j][k];
			}
		}
	}
	/*for (i = 0; i < column; i++) {
		printf("weights[%d]=%f\n",i, weights[i]);
	}*/
}

double predict(int col,double *array, double *weights) {//预测某一行的值
	double activation = weights[0];
	int i;
	for (i = 0; i < col - 1; i++)
		activation += weights[i + 1] * array[i];
	double output = 0.0;
	if (activation >= 0.0)
		output = 1.0;
	else
		output = 0.0;
	return output;
}