## $k$-Nearest Neighbors

> $k$-近邻（$k$-Nearest Neighbour），简称KNN。KNN算法最初由Cover和Hart于1968年提出，是一个理论上比较成熟的方法，也是最简单的机器学习算法之一。

### 1.算法介绍

KNN算法是一种监督学习算法。KNN算法的核心就是从训练集中选取 $k$ 个与新样本**相似度最高**的样本（ $k$ 个近邻），通过这 $k$ 个近邻的类别来确定待新样本的类别。其中，$k$ 的大小是可以自由选取的。

如何衡量样本之间的相似度呢？下面，引入欧式距离公式：

我们知道，两点 $(x_0,y_0),(x_1,y_1)$ 之间的欧几里得距离公式如下：
$$
d = \sqrt{(x_0-x_1)^2+(y_0-y_1)^2}
$$
然而，用于机器学习的样本可能是高维的。因此，我们将其推广至 $n$ 维，设两点分别是$(x_1^{(1)},x_2^{(1)},\dots,x_n^{(1)})$与$(x_1^{(2)},x_2^{(2)},\dots,x_n^{(2)})$，则两点之间的欧几里得距离公式为：
$$
d = \sqrt{\sum_{i=1}^n(x_{i}^{(1)}-x_{i}^{(2)})^2}
$$

若训练集共m个样本，我们得到每一个训练样本与新样本的距离序列 $\{d_i\}(i=1,2,\dots,m)$。

**我们认为两个样本的距离最小时，它们最相似。**因此我们从序列 $\{d_i\}$ 中选取最小的 $k$ 个样本，设它们的类别分别是  $y_1,y_2,\dots,y_k$ ，我们求出这 $k$ 个数的众数 $l$ ，$l$ 即为KNN算法对新样本的分类。

**当然，KNN算法不仅可用于分类任务，也可以用于回归任务。当KNN算法用于归回任务时，我们可以求出新样本的 $k$ 个近邻类别的均值 $m$，$m$ 即为KNN算法对新样本的预测。**

### 2.算法讲解

#### 2.1  欧几里得距离

实现KNN算法的第一步是计算同一份数据集中的任意两行数据的距离。

下面给出代码，完成欧几里得距离的计算。

```c
#include<stdio.h>
#include<math.h>

double euclidean_distance(double *row1, double *row2, int col) {
	double distance = 0;
	for (int i = 0; i < col - 1; i++) {
		distance += pow((row1[i] - row2[i]), 2);
	}
	return sqrt(distance);
}

void main(){
    double test_data[10][3] = {
        {2.56373457, 2.63727045, 0},
        {1.62548536, 2.26342507, 0},
        {3.69634668, 4.34629352, 0},
        {1.45607019, 1.84562031, 0},
        {3.06407232, 3.00530597, 0},
        {7.54753121, 2.98926223, 1},
        {5.12422124, 2.08862677, 1},
        {6.86549671, 1.77106367, 1},
        {8.67541865, -0.24206865, 1},
        {7.67375646, 3.76356301, 1}
    };
    double** dataset;
	dataset = (double**)malloc(10 * sizeof(double*));
	for (int i = 0; i < 10; ++i) {
		dataset[i] = test_data[i];
	}
    double result;
    for (int i = 0; i < 10; i++) {
        result = euclidean_distance(test_data[0], test_data[i], 3);
        printf("%f\n", result);
    }
}
```

运行代码，得到每一行数据与第一行数据的距离：

```c
0.000000
1.009986
2.050261
1.361481
0.621118
4.996211
2.618607
4.388106
6.755981
5.232672
```

#### 2.2  获取近邻

通过计算某一条数据$x_i$与其他所有数据之间的距离，得到一个距离集合。将该距离集合中的元素由小到大排序，找到与$x_i$最近的k个数据，即k个近邻，返回他们的标签。

下面以3个近邻为例，沿用上面的数据，求出与$x_0$距离最近的3条数据的标签。

```c
void QuickSort(double **arr, int L, int R) {
	int i = L;
	int j = R;
	int kk = (L + R) / 2; //支点
	double pivot = arr[kk][0];
	//左右两端进行扫描，直到两端交替
	while (i <= j) {
        //寻找比支点大的数
		while (pivot > arr[i][0])
		{
			i++;
		}//寻找比支点小的数
		while (pivot < arr[j][0])
		{
			j--;
		}//此时已经分别找到了比支点小的数(右边)、比支点大的数(左边)，交换他们
		if (i <= j) {
			double *temp = arr[i];
			arr[i] = arr[j];
			arr[j] = temp;
			i++; j--;
		}
	}//上面的while保证了第一次排序支点的左边比支点小，支点的右边比支点大了。
	//左边再做排序，直到左边剩下一个数(递归出口)
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

void main(){
    double result;
    for (int i = 0; i < 10; i++) {
        result = euclidean_distance(dataset[0], dataset[i], 3);
        printf("%f\n", result);
    }
	int num_neighbors = 3;
	double* neighbors = get_neighbors(dataset, 10, 3, dataset[0], num_neighbors);
	for (int i = 0; i < num_neighbors; i++) {
		printf("%f\n", neighbors[i]);
	}
}
```
得到近邻的标签为：

```c
0.000000
0.000000
0.000000
```

#### 2.3  预测结果

```c
double predict(double **train_data, int train_row, int col, double *test_row, int num_neighbors) {
	double* neighbors = get_neighbors(train_data, train_row, col, test_row, num_neighbors);
	double result = 0;
	for (int i = 0; i < num_neighbors; i++) {
		result += neighbors[i];
	}
	return result / num_neighbors;
}

void main() {

	double test_data[10][3] = {
		{2.56373457, 2.63727045, 0},
		{1.62548536, 2.26342507, 0},
		{3.69634668, 4.34629352, 0},
		{1.45607019, 1.84562031, 0},
		{3.06407232, 3.00530597, 0},
		{7.54753121, 2.98926223, 1},
		{5.12422124, 2.08862677, 1},
		{6.86549671, 1.77106367, 1},
		{8.67541865, -0.24206865, 1},
		{7.67375646, 3.76356301, 1}
	};
	double** dataset;
	dataset = (double**)malloc(10 * sizeof(double*));
	for (int i = 0; i < 10; ++i) {
		dataset[i] = test_data[i];
	}

	int num_neighbors = 3;
	double result;
	result = predict(dataset, 10, 3, dataset[0], num_neighbors);
	printf("%f\n", result);
}
```

该数据的标签为0，而结果为：

```c
0
```

### 3.算法实现步骤

下将通过通过多C文件，以[鲍鱼数据集](https://goo.gl/BDYgh5)为例，使用KNN算法，回归预测鲍鱼的环数量。

#### 3.1  read_csv.c

该步骤代码与前面代码一致，不再重复给出。

#### 3.2  normalize.c

该步骤代码与前面代码一致，不再重复给出。

#### 3.3  k_fold.c

该步骤代码与前面代码一致，不再重复给出。

#### 3.4  test_prediction.c

```c
#include<stdlib.h>
#include<stdio.h>
#include<math.h>

extern void QuickSort(double **arr, int L, int R);
extern double euclidean_distance(double *row1, double *row2, int col);
extern double* get_neighbors(double **train_data, int train_row, int col, double *test_row, int num_neighbors);
extern double predict(double **train_data, int train_row, int col, double *test_row, int num_neighbors);

double* get_test_prediction(double **train, int train_size,  double **test, int test_size, int col, int num_neighbors)
{
	double* predictions = (double*)malloc(test_size * sizeof(double));
	for (int i = 0; i < test_size; i++)
	{
		predictions[i] = predict(train, train_size,col,test[i],num_neighbors);
	}
	return predictions;//返回对test的预测数组
}
```

#### 3.5  rmse.c

该步骤代码与前面代码一致，不再重复给出。

#### 3.6  evaluate.c

```c
#include<stdlib.h>
#include<stdio.h>

extern double* get_test_prediction(double **train, int train_size, double **test, int test_size, int col, int num_neighbors);
extern double rmse_metric(double *actual, double *predicted, int fold_size);
extern double***  cross_validation_split(double **dataset, int row, int col, int n_folds, int fold_size);

void evaluate_algorithm(double **dataset, int row, int col, int n_folds, int num_neighbors) {
	int fold_size = (int)row / n_folds;
	double ***split = cross_validation_split(dataset, row, n_folds, fold_size, col);
	int i, j, k, l;
	int test_size = fold_size;
	int train_size = fold_size * (n_folds - 1);
	double* score = (double*)malloc(n_folds * sizeof(double));

	for (i = 0; i < n_folds; i++) {
		//因为要遍历删除，所以拷贝一份split
		double*** split_copy = (double***)malloc(n_folds * sizeof(double**));
		for (j = 0; j < n_folds; j++) {
			split_copy[j] = (double**)malloc(fold_size * sizeof(double*));
			for (k = 0; k < fold_size; k++) {
				split_copy[j][k] = (double*)malloc(col * sizeof(double));
			}
		}
		for (j = 0; j < n_folds; j++)
		{
			for (k = 0; k < fold_size; k++)
			{
				for (l = 0; l < col; l++)
				{
					split_copy[j][k][l] = split[j][k][l];
				}
			}
		}
		double** test_set = (double**)malloc(test_size * sizeof(double*));
		for (j = 0; j < test_size; j++) {
			test_set[j] = (double*)malloc(col * sizeof(double));
			for (k = 0; k < col; k++) {
				test_set[j][k] = split_copy[i][j][k];
			}
		}
		for (j = i; j < n_folds - 1; j++) {
			split_copy[j] = split_copy[j + 1];
		}//删除取出来的fold
		double** train_set = (double**)malloc(train_size * sizeof(double*));
		for (k = 0; k < n_folds - 1; k++) {
			for (l = 0; l < fold_size; l++) {
				train_set[k*fold_size + l] = (double*)malloc(col * sizeof(double));
				train_set[k*fold_size + l] = split_copy[k][l];
			}
		}
		double* predicted = (double*)malloc(test_size * sizeof(double));
		predicted = get_test_prediction(train_set, train_size, test_set,test_size, col, num_neighbors);
		double* actual = (double*)malloc(test_size * sizeof(double));
		for (l = 0; l < test_size; l++) {
			actual[l] = test_set[l][col - 1];
		}
		double rmse = rmse_metric(actual, predicted, test_size);
		score[i] = rmse;
		printf("rmse[%d]=%f\n", i, score[i]);
		free(split_copy);
	}
	double total = 0;
	for (l = 0; l < n_folds; l++) {
		total += score[l];
	}
	printf("mean_rmse=%f\%\n", total / n_folds);
}
```

#### 3.7 main.c

```c
#define _CRT_SECURE_NO_WARNINGS
#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<math.h>

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern void normalize_dataset(double **dataset, int row, int col);
extern void evaluate_algorithm(double **dataset, int row, int col, int n_folds, int num_neighbors);

void QuickSort(double **arr, int L, int R);
double euclidean_distance(double *row1, double *row2, int col);
double* get_neighbors(double **train_data, int train_row, int col, double *test_row, int num_neighbors);
double predict(double **train_data, int train_row, int col, double *test_row, int num_neighbors);

void main() {
	char filename[] = "abalone.csv";
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
	normalize_dataset(dataset, row, col);
	int k_fold = 5;
	int num_neighbors = 5;
	evaluate_algorithm(dataset, row, col, k_fold, num_neighbors);
}

void QuickSort(double **arr, int L, int R) {
	int i = L;
	int j = R;
	//支点
	int kk = (L + R) / 2;
	double pivot = arr[kk][0];
	//左右两端进行扫描，只要两端还没有交替，就一直扫描
	while (i <= j) {//寻找直到比支点大的数
		while (pivot > arr[i][0])
		{
			i++;
		}//寻找直到比支点小的数
		while (pivot < arr[j][0])
		{
			j--;
		}//此时已经分别找到了比支点小的数(右边)、比支点大的数(左边)，它们进行交换
		if (i <= j) {
			double *temp = arr[i];
			arr[i] = arr[j];
			arr[j] = temp;
			i++; j--;
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

// Calculate the Euclidean distance between two vectors
double euclidean_distance(double *row1, double *row2, int col) {
	double distance = 0;
	for (int i = 0; i < col - 1; i++) {
		distance += pow((row1[i] - row2[i]), 2);
	}
	return distance;
}

// Locate the most similar neighbors
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

double predict(double **train_data, int train_row, int col, double *test_row, int num_neighbors) {
	double* neighbors = get_neighbors(train_data, train_row, col, test_row, num_neighbors);
	double result = 0;
	for (int i = 0; i < num_neighbors; i++) {
		result += neighbors[i];
	}
	return result / num_neighbors;
}

```

