#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<math.h>

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern void evaluate_algorithm(double **dataset, int row, int col, int n_folds);

void quicksort(double *arr, int L, int R) {
	int i = L;
	int j = R;
	//支点
	int kk = (L + R) / 2;
	double pivot = arr[kk];
	//左右两端进行扫描，只要两端还没有交替，就一直扫描
	while (i <= j) {//寻找直到比支点大的数
		while (pivot > arr[i])
		{
			i++;
		}//寻找直到比支点小的数
		while (pivot < arr[j])
		{
			j--;
		}//此时已经分别找到了比支点小的数(右边)、比支点大的数(左边)，它们进行交换
		if (i <= j) {
			double temp = arr[i];
			arr[i] = arr[j];
			arr[j] = temp;
			i++; j--;
		}
	}//上面一个while保证了第一趟排序支点的左边比支点小，支点的右边比支点大了。
	//“左边”再做排序，直到左边剩下一个数(递归出口)
	if (L < j)
	{
		quicksort(arr, L, j);
	}
	//“右边”再做排序，直到右边剩下一个数(递归出口)
	if (i < R)
	{
		quicksort(arr, i, R);
	}
}
double get_mean(double**dataset, int row, int col) {
	int i;
	double mean = 0;
	for (i = 0; i < row; i++) {
		mean += dataset[i][col];
	}
	return mean / row;
}
double get_std(double**dataset, int row, int col) {
	int i;
	double mean = 0;
	double std = 0;
	for (i = 0; i < row; i++) {
		mean += dataset[i][col];
	}
	mean /= row;
	for (i = 0; i < row; i++) {
		std += pow((dataset[i][col]-mean),2);
	}
	return sqrt(std / (row - 1));
}

int get_class_num(double **dataset,int row, int col) {
	int i;
	int num = 1;
	double *class_data = (double *)malloc(row * sizeof(double));
	for (i = 0; i < row; i++) {
		class_data[i] = dataset[i][col - 1];
	}
	quicksort(class_data, 0, row - 1);
	for (i = 0; i < row-1; i++) {
		if (class_data[i] != class_data[i + 1]) {
			num += 1;
		}
	}
	return num;
}
int *get_class_num_list(double **dataset, int class_num, int row, int col) {
	int i,j;
	int *class_num_list = (int *)malloc(class_num * sizeof(int));
	for (j = 0; j < class_num; j++) {
		class_num_list[j] = 0;
	}
	for (j = 0; j < class_num; j++) {
		for (i = 0; i < row; i++) {
			if (dataset[i][col - 1] == j) {
				class_num_list[j] += 1;
			}
		}
	}
	return class_num_list;
}

double ***separate_by_class(double **dataset,int class_num, int *class_num_list, int row, int col) {
	double ***separated;
	separated = (double***)malloc(class_num * sizeof(double**));
	int i, j;
	for (i = 0; i < class_num; i++) {
		separated[i] = (double**)malloc(class_num_list[i] * sizeof(double *));
		for (j = 0; j < class_num_list[i]; j++) {
			separated[i][j] = (double*)malloc(col * sizeof(double));
		}
	}
	int* index = (int *)malloc(class_num * sizeof(int));
	for (i = 0; i < class_num; i++) {
		index[i] = 0;
	}
	for (i = 0; i < row; i++) {
		for (j = 0; j < class_num; j++) {
			if (dataset[i][col - 1] == j) {
				separated[j][index[j]] = dataset[i];
				index[j]++;
			}
		}
	}
	return separated;
}
double **summarize_dataset(double **dataset,int row, int col) {
	int i;
	double **summary = (double**)malloc((col - 1) * sizeof(double *));
	for (i = 0; i < (col - 1); i++) {
		summary[i] = (double*)malloc(2 * sizeof(double));
		summary[i][0] = get_mean(dataset, row, i);
		summary[i][1] = get_std(dataset, row, i);		
	}
	return summary;
}
double ***summarize_by_class(double **train, int class_num, int *class_num_list, int row, int col) {
	int i;
	double ***summarize;
	summarize = (double***)malloc(class_num * sizeof(double**));
	double ***separate = separate_by_class(train, class_num, class_num_list, row, col);
	for (i = 0; i < class_num; i++) {
		summarize[i] = summarize_dataset(separate[i], class_num_list[i], col);
	}
	return summarize;
}

double calculate_probability(double x, double mean, double std)
{
	double pi = acos(-1.0);
	double p = 1 / (pow(2 * pi, 0.5) * std) * 
		exp(-(pow((x - mean), 2) / (2 * pow(std, 2))));
	return p;
}
double *calculate_class_probabilities(double ***summaries,double *test_row, int class_num, int *class_num_list, int row, int col) {
	int i, j;
	double *probabilities = (double *)malloc(class_num * sizeof(double));
	for (i = 0; i < class_num; i++) {
		probabilities[i] = (double)class_num_list[i] / row;
	}
	for (i = 0; i < class_num; i++) {
		for (j = 0; j < col-1; j++) {
			probabilities[i] *= calculate_probability(test_row[j], summaries[i][j][0], summaries[i][j][1]);
		}
	}
	return probabilities;
}
double predict(double ***summaries, double *test_row, int class_num, int *class_num_list, int row, int col) {
	int i;
	double *probabilities = calculate_class_probabilities(summaries, test_row, class_num, class_num_list, row, col);
	double label = 0;
	double best_prob = probabilities[0];
	for (i = 1; i < class_num; i++) {
		if (probabilities[i] > best_prob) {
			label = i;
			best_prob = probabilities[i];
		}
	}
	return label;
}

void main() {
	char filename[] = "iris.csv";
	char line[1024];
	int row = get_row(filename);
	int col = get_col(filename);
	//printf("row = %d, col = %d\n", row, col);
	double **dataset;
	dataset = (double **)malloc(row * sizeof(double *));
	for (int i = 0; i < row; ++i) {
		dataset[i] = (double *)malloc(col * sizeof(double));
	}
	get_two_dimension(line, dataset, filename);
	int n_folds = 5;
	evaluate_algorithm(dataset, row, col, n_folds);
}

