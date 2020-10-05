#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<math.h>


void QuickSort(double **arr, int L, int R) {
	int i = L;
	int j = R;
	//支点
	int kk = (L + R) / 2;
	double pivot = arr[kk][0];
	//左右两端进行扫描，只要两端还没有交替，就一直扫描
	while (i <= j) {
		//寻找直到比支点大的数
		while (pivot > arr[i][0])
		{
			i++;
		}
		//寻找直到比支点小的数
		while (pivot < arr[j][0])
		{
			j--;
		}
		//此时已经分别找到了比支点小的数(右边)、比支点大的数(左边)，它们进行交换
		if (i <= j) {
			double *temp = arr[i];
			arr[i] = arr[j];
			arr[j] = temp;
			//double temp2 = arr[i][1];
			//arr[i][1] = arr[j][1];
			//arr[j][1] = temp2;
			i++;
			j--;
		}
	}//上面一个while保证了第一趟排序支点的左边比支点小，支点的右边比支点大了。
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
double euclidean_distance(double *row1, double *row2, int col) {
	double distance = 0;
	for (int i = 0; i < col - 1; i++) {
		distance += pow((row1[i] - row2[i]), 2);
	}
	return distance;
}
double* get_neighbors(double **train_data, int train_row, int col, double *test_row, int num_neighbors) {
	double *neighbors = (double *)malloc(num_neighbors * sizeof(double));
	double **distances = (double **)malloc(train_row * sizeof(double *));
	for (int i = 0; i < train_row; i++) {
		distances[i] = (double *)malloc(2 * sizeof(double));
		distances[i][0] = euclidean_distance(train_data[i], test_row, col);
		distances[i][1] = train_data[i][col - 1];
	}
	QuickSort(distances, 0, train_row - 1);
	for (int i = 0; i < num_neighbors; i++) {
		neighbors[i] = distances[i][1];
	}
	return neighbors;
}
double knn_single_predict(double **train_data, int train_row, int col, double *test_row, int num_neighbors) {
	double* neighbors = get_neighbors(train_data, train_row, col, test_row, num_neighbors);
	double result = 0;
	for (int i = 0; i < num_neighbors; i++) {
		result += neighbors[i];
	}
	return round(result / num_neighbors);
}


double* knn_predict(double **train, int train_size, double **test, int test_size, int col, int num_neighbors)
{
	double* predictions = (double*)malloc(test_size * sizeof(double));
	for (int i = 0; i < test_size; i++)
	{
		predictions[i] = knn_single_predict(train, train_size, col, test[i], num_neighbors);
	}
	return predictions;//返回对test的预测数组
}

