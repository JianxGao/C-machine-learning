#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern void normalize_dataset(double **dataset, int row, int col);
extern void evaluate_algorithm(double **dataset, int row, int col, int n_folds, int num_neighbors);

void QuickSort(double **arr, int L, int R)
{
	int i = L;
	int j = R;
	//支点
	int kk = (L + R) / 2;
	double pivot = arr[kk][0];
	//左右两端进行扫描，只要两端还没有交替，就一直扫描
	while (i <= j)
	{ //寻找直到比支点大的数
		while (pivot > arr[i][0])
		{
			i++;
		} //寻找直到比支点小的数
		while (pivot < arr[j][0])
		{
			j--;
		} //此时已经分别找到了比支点小的数(右边)、比支点大的数(左边)，它们进行交换
		if (i <= j)
		{
			double *temp = arr[i];
			arr[i] = arr[j];
			arr[j] = temp;
			i++;
			j--;
		}
	} //上面一个while保证了第一趟排序支点的左边比支点小，支点的右边比支点大了。
	//“左边”再做排序，直到左边剩下一个数(递归出口)
	if (L < j)
	{
		QuickSort(arr, L, j);
	}
	//“右边”再做排序，直到右边剩下一个数(递归出口)
	if (i < R)
	{
		QuickSort(arr, i, R);
	}
}
// Calculate the Euclidean distance between two vectors
double euclidean_distance(double *row1, double *row2, int col)
{
	double distance = 0;
	for (int i = 0; i < col - 1; i++)
	{
		distance += pow((row1[i] - row2[i]), 2);
	}
	return sqrt(distance);
}
// Locate the most similar neighbors
double *get_neighbors(double **train_data, int train_row, int col, double *test_row, int num_neighbors)
{
	double *neighbors = (double *)malloc(num_neighbors * sizeof(double));
	double **distances = (double **)malloc(train_row * sizeof(double *));
	for (int i = 0; i < train_row; i++)
	{
		distances[i] = (double *)malloc(2 * sizeof(double));
		distances[i][0] = euclidean_distance(train_data[i], test_row, col);
		distances[i][1] = train_data[i][col - 1];
	}
	QuickSort(distances, 0, train_row - 1);
	for (int i = 0; i < num_neighbors; i++)
	{
		neighbors[i] = distances[i][1];
	}
	return neighbors;
}
double predict(double **train_data, int train_row, int col, double *test_row, int num_neighbors)
{
	double *neighbors = get_neighbors(train_data, train_row, col, test_row, num_neighbors);
	double result = 0;
	for (int i = 0; i < num_neighbors; i++)
	{
		result += neighbors[i];
	}
	return result / num_neighbors;
}

void main()
{
	char filename[] = "abalone.csv";
	char line[1024];
	int row = get_row(filename);
	int col = get_col(filename);
	//printf("row = %d, col = %d\n", row, col);
	double **dataset;
	dataset = (double **)malloc(row * sizeof(double *));
	for (int i = 0; i < row; ++i)
	{
		dataset[i] = (double *)malloc(col * sizeof(double));
	}
	get_two_dimension(line, dataset, filename);
	normalize_dataset(dataset, row, col);
	int k_fold = 5;
	int num_neighbors = 5;
	evaluate_algorithm(dataset, row, col, k_fold, num_neighbors);
}