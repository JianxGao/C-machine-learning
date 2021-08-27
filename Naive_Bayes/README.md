## 3.6 Naive Bayes

> 朴素贝叶斯算法，是应用最为广泛的分类算法之一。该算法利用贝叶斯定理与特征条件独立假设做预测，直接且易于理解。该算法在实际运用中，往往能得到意想不到的好结果。

### 3.6.1 算法介绍

朴素贝叶斯算法的其本质就是计算$P(class|data)$，即数据$data$属于某一类别$class$的概率。

朴素贝叶斯算法的核心就是贝叶斯公式，贝叶斯公式为我们提供了一个适用于计算一些数据属于某一类别的概率的计算方法。贝叶斯公式如下：

$$
P(class|data) = \dfrac{P(data|class)\times P(class)}{P(data)}
$$

其中，$P(class|data)$表示$data$属于某个$class$的概率。同时，上式假设各个特征条件是独立的。

我们认为，使得$P(class|data)$最大的$class$就是该$data$所属的$class$，从而我们可以预测出该数据所属类别。

下面，我们将结合代码讲解，朴素贝叶斯算法是如何计算$P(class|data)$，进而对$data$做预测的。

### 3.6.2 算法讲解

#### 数据分类

我们需把所有的数据按照各自的类别进行分类，组成一个新的数组。下面先给出将数据的分类函数：

```c
double*** separate_by_class(double **dataset,int class_num, int *class_num_list, int row, int col) {
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
```

以下面的10条数据为例，利用上述函数对数据进行分类

```c
X1				X2				Lable
2.000000        2.000000        0.000000
2.005000        1.995000        0.000000
2.010000        1.990000        0.000000
2.015000        1.985000        0.000000
2.020000        1.980000        0.000000

5.000000        5.000000        1.000000
5.005000        4.995000        1.000000
5.010000        4.990000        1.000000
5.015000        4.985000        1.000000
5.020000        4.980000        1.000000
```

代码如下：

```c
double*** separate_by_class(double **dataset,int class_num, int *class_num_list, int row, int col) {
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

void main() {
    double **dataset;
    dataset = (double **)malloc(row * sizeof(double *));
    for (int i = 0; i < row; ++i) {
        dataset[i] = (double *)malloc(col * sizeof(double));
    }
    for (int i = 0; i < 5; i++) {
        dataset[i][0] = 2 + i * 0.005;
        dataset[i][1] = 2 - i * 0.005;
        dataset[i][2] = 0;
    }
    for (int i = 0; i < 5; i++) {
        dataset[i+5][0] = 5 + i * 0.005;
        dataset[i+5][1] = 5 - i * 0.005;
        dataset[i + 5][2] = 1;
    }
    int class_num_list[2] = {5,5};
    double ***separated = separate_by_class(dataset, 2, class_num_list, 10, 3);
    // 输出结果
    for (int i = 0; i < 2; i++) {
        //先按照类别输出
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 3; k++) {
                printf("%f\t", separated[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
```

输出结果如下：

```c
2.000000        2.000000        0.000000
2.005000        1.995000        0.000000
2.010000        1.990000        0.000000
2.015000        1.985000        0.000000
2.020000        1.980000        0.000000

5.000000        5.000000        1.000000
5.005000        4.995000        1.000000
5.010000        4.990000        1.000000
5.015000        4.985000        1.000000
5.020000        4.980000        1.000000
```

#### 计算统计量

我们需要计算数据的均值与标准差，其公式与前文中提到的一致，如下所示：

$mean = \dfrac{\sum_{i=1}^nx_i}{count(x)}$

$standard\;\;deviation = \sqrt{\dfrac{\sum_{i-1}^n{(x_i-mean(x))^2}}{count(x)-1}}$

其代码如下：

```c
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
```

仍然以下面的10条数据为例，利用上述函数按照类别计算数据的统计量

```c
X1				X2				Lable
2.000000        2.000000        0.000000
2.005000        1.995000        0.000000
2.010000        1.990000        0.000000
2.015000        1.985000        0.000000
2.020000        1.980000        0.000000

5.000000        5.000000        1.000000
5.005000        4.995000        1.000000
5.010000        4.990000        1.000000
5.015000        4.985000        1.000000
5.020000        4.980000        1.000000
```

代码如下：

```c
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

double*** separate_by_class(double **dataset,int class_num, int *class_num_list, int row, int col) {
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

double** summarize_dataset(double **dataset,int row, int col) {
    int i;
    double **summary = (double**)malloc((col - 1) * sizeof(double *));
    for (i = 0; i < (col - 1); i++) {
        summary[i] = (double*)malloc(2 * sizeof(double));
        summary[i][0] = get_mean(dataset, row, i);
        summary[i][1] = get_std(dataset, row, i);
    }
    return summary;
}

double*** summarize_by_class(double **train, int class_num, int *class_num_list, int row, int col) {
    int i;
    double ***summarize;
    summarize = (double***)malloc(class_num * sizeof(double**));
    double ***separate = separate_by_class(train, class_num, class_num_list, row, col);
    for (i = 0; i < class_num; i++) {
        summarize[i] = summarize_dataset(separate[i], class_num_list[i], col);
    }
    return summarize;
}

void main() {
    int row = 10;
    int col = 3;
    int class_num = 2;
    int class_num_list[2] = { 5,5 };
    double** dataset;
    dataset = (double**)malloc(row * sizeof(double*));
    for (int i = 0; i < row; ++i) {
        dataset[i] = (double*)malloc(col * sizeof(double));
    }
    for (int i = 0; i < 5; i++) {
        dataset[i][0] = 2 + i * 0.005;
        dataset[i][1] = 2 - i * 0.005;
        dataset[i][2] = 0;
    }
    for (int i = 0; i < 5; i++) {
        dataset[i + 5][0] = 5 + i * 0.005;
        dataset[i + 5][1] = 5 - i * 0.005;
        dataset[i + 5][2] = 1;
    }
    double*** summarize = summarize_by_class(dataset, class_num, class_num_list, row, col);
    for (int i = 0; i < 2; i++) {
        //先按照类别输出
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                printf("%f\t", summarize[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
```

按照类别依次得到每列数据的均值与方差：

```c
2.010000        0.007906
1.990000        0.007906

5.010000        0.007906
4.990000        0.007906
```

#### 高斯概率分布函数

高斯概率密度函数表达式为：

$$
probability(x) = \dfrac{e^{-\dfrac{(x-mean(x))^2}{2\times standard\_{} deviation^2}}}{\sqrt{2\times PI}\times standard\_{} deviation}
$$

计算代码如下：

```c
double calculate_probability(double x, double mean, double std)
{
    double pi = acos(-1.0);
    double p = 1 / (pow(2 * pi, 0.5) * std) * exp(-(pow((x - mean), 2) / (2 * pow(std, 2))));
    return p;
}
```

#### 类别概率

下面就是朴素贝叶斯的关键——计算数据属于某一类别的概率。

代码如下：

```c
double* calculate_class_probabilities(double ***summaries,double *test_row, int class_num, int *class_num_list, int row, int col) {
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
```

### 3.6.3 算法代码

我们现在知道了如何实现**朴素贝叶斯算法**，那么我们把它应用到[鸢尾花数据集 iris.csv](https://aistudio.baidu.com/aistudio/datasetdetail/105756/0)

我们给出链接：https://aistudio.baidu.com/aistudio/datasetdetail/105756/0

#### C语言细节讲解

本节假设您已下载数据集 `iris.csv`，并且它在当前工作目录中可用。下面我们给出一个完整实例，使用C语言详细讲解每一处细节。我们给出每一个.c文件的所有代码：

##### 1) read_csv.c

该文件代码与前面代码一致，不再重复给出。

##### 2) normalize.c

该文件代码与前面代码一致，不再重复给出。

##### 3) k_fold.c

该文件代码与前面代码一致，不再重复给出。

##### 4) score.c

该文件代码与前面代码一致，不再重复给出。

##### 5) test_prediction.c

```c
#include <stdlib.h>
#include <stdio.h>

extern double predict(double ***summaries, double *test_row, int class_num, int *class_num_list, int row, int col);
extern int get_class_num(double **dataset, int row, int col);
extern int *get_class_num_list(double **dataset, int class_num, int row, int col);
extern double ***summarize_by_class(double **train, int class_num, int *class_num_list, int row, int col);

double *get_test_prediction(double **train, int train_size, double **test, int test_size, int col)
{
	int class_num = get_class_num(train, train_size, col);
	int *class_num_list = get_class_num_list(train, class_num, train_size, col);
	double *predictions = (double *)malloc(test_size * sizeof(double)); //因为test_size和fold_size一样大
	double ***summaries = summarize_by_class(train, class_num, class_num_list, train_size, col);
	for (int i = 0; i < test_size; i++)
	{
		predictions[i] = predict(summaries, test[i], class_num, class_num_list, train_size, col);
	}
	return predictions; //返回对test的预测数组
}
```

##### 6) evaluate.c

```c
#include <stdlib.h>
#include <stdio.h>

extern double *get_test_prediction(double **train, int train_size, double **test, int test_size, int col);
extern double accuracy_metric(double *actual, double *predicted, int fold_size);
extern double ***cross_validation_split(double **dataset, int row, int col, int n_folds, int fold_size);

void evaluate_algorithm(double **dataset, int row, int col, int n_folds)
{
	int fold_size = (int)row / n_folds;
	double ***split = cross_validation_split(dataset, row, n_folds, fold_size, col);
	int i, j, k, l;
	int test_size = fold_size;
	int train_size = fold_size * (n_folds - 1);
	double *score = (double *)malloc(n_folds * sizeof(double));

	for (i = 0; i < n_folds; i++)
	{
		double ***split_copy = (double ***)malloc(n_folds * sizeof(double **));
		for (j = 0; j < n_folds; j++)
		{
			split_copy[j] = (double **)malloc(fold_size * sizeof(double *));
			for (k = 0; k < fold_size; k++)
			{
				split_copy[j][k] = (double *)malloc(col * sizeof(double));
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
		double **test_set = (double **)malloc(test_size * sizeof(double *));
		for (j = 0; j < test_size; j++)
		{
			test_set[j] = (double *)malloc(col * sizeof(double));
			for (k = 0; k < col; k++)
			{
				test_set[j][k] = split_copy[i][j][k];
			}
		}
		for (j = i; j < n_folds - 1; j++)
		{
			split_copy[j] = split_copy[j + 1];
		}
		double **train_set = (double **)malloc(train_size * sizeof(double *));
		for (k = 0; k < n_folds - 1; k++)
		{
			for (l = 0; l < fold_size; l++)
			{
				train_set[k * fold_size + l] = (double *)malloc(col * sizeof(double));
				train_set[k * fold_size + l] = split_copy[k][l];
			}
		}
		double *predicted = (double *)malloc(test_size * sizeof(double));
		predicted = get_test_prediction(train_set, train_size, test_set, test_size, col);
		double *actual = (double *)malloc(test_size * sizeof(double));
		for (l = 0; l < test_size; l++)
		{
			actual[l] = test_set[l][col - 1];
		}

		double acc = accuracy_metric(actual, predicted, test_size);
		score[i] = acc;
		printf("Scores[%d] = %f%%\n", i, score[i]);
		free(split_copy);
	}
	double total = 0;
	for (l = 0; l < n_folds; l++)
	{
		total += score[l];
	}
	printf("mean_accuracy = %f%%\n", total / n_folds);
}
```

##### 7) main.c

```c
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern void evaluate_algorithm(double **dataset, int row, int col, int n_folds);

void quicksort(double *arr, int L, int R)
{
	int i = L;
	int j = R;
	//支点
	int kk = (L + R) / 2;
	double pivot = arr[kk];
	//左右两端进行扫描，只要两端还没有交替，就一直扫描
	while (i <= j)
	{ //寻找直到比支点大的数
		while (pivot > arr[i])
		{
			i++;
		} //寻找直到比支点小的数
		while (pivot < arr[j])
		{
			j--;
		} //此时已经分别找到了比支点小的数(右边)、比支点大的数(左边)，它们进行交换
		if (i <= j)
		{
			double temp = arr[i];
			arr[i] = arr[j];
			arr[j] = temp;
			i++;
			j--;
		}
	} //上面一个while保证了第一趟排序支点的左边比支点小，支点的右边比支点大了。
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
double get_mean(double **dataset, int row, int col)
{
	int i;
	double mean = 0;
	for (i = 0; i < row; i++)
	{
		mean += dataset[i][col];
	}
	return mean / row;
}
double get_std(double **dataset, int row, int col)
{
	int i;
	double mean = 0;
	double std = 0;
	for (i = 0; i < row; i++)
	{
		mean += dataset[i][col];
	}
	mean /= row;
	for (i = 0; i < row; i++)
	{
		std += pow((dataset[i][col] - mean), 2);
	}
	return sqrt(std / (row - 1));
}

int get_class_num(double **dataset, int row, int col)
{
	int i;
	int num = 1;
	double *class_data = (double *)malloc(row * sizeof(double));
	for (i = 0; i < row; i++)
	{
		class_data[i] = dataset[i][col - 1];
	}
	quicksort(class_data, 0, row - 1);
	for (i = 0; i < row - 1; i++)
	{
		if (class_data[i] != class_data[i + 1])
		{
			num += 1;
		}
	}
	return num;
}
int *get_class_num_list(double **dataset, int class_num, int row, int col)
{
	int i, j;
	int *class_num_list = (int *)malloc(class_num * sizeof(int));
	for (j = 0; j < class_num; j++)
	{
		class_num_list[j] = 0;
	}
	for (j = 0; j < class_num; j++)
	{
		for (i = 0; i < row; i++)
		{
			if (dataset[i][col - 1] == j)
			{
				class_num_list[j] += 1;
			}
		}
	}
	return class_num_list;
}

double ***separate_by_class(double **dataset, int class_num, int *class_num_list, int row, int col)
{
	double ***separated;
	separated = (double ***)malloc(class_num * sizeof(double **));
	int i, j;
	for (i = 0; i < class_num; i++)
	{
		separated[i] = (double **)malloc(class_num_list[i] * sizeof(double *));
		for (j = 0; j < class_num_list[i]; j++)
		{
			separated[i][j] = (double *)malloc(col * sizeof(double));
		}
	}
	int *index = (int *)malloc(class_num * sizeof(int));
	for (i = 0; i < class_num; i++)
	{
		index[i] = 0;
	}
	for (i = 0; i < row; i++)
	{
		for (j = 0; j < class_num; j++)
		{
			if (dataset[i][col - 1] == j)
			{
				separated[j][index[j]] = dataset[i];
				index[j]++;
			}
		}
	}
	return separated;
}
double **summarize_dataset(double **dataset, int row, int col)
{
	int i;
	double **summary = (double **)malloc((col - 1) * sizeof(double *));
	for (i = 0; i < (col - 1); i++)
	{
		summary[i] = (double *)malloc(2 * sizeof(double));
		summary[i][0] = get_mean(dataset, row, i);
		summary[i][1] = get_std(dataset, row, i);
	}
	return summary;
}
double ***summarize_by_class(double **train, int class_num, int *class_num_list, int row, int col)
{
	int i;
	double ***summarize;
	summarize = (double ***)malloc(class_num * sizeof(double **));
	double ***separate = separate_by_class(train, class_num, class_num_list, row, col);
	for (i = 0; i < class_num; i++)
	{
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
double *calculate_class_probabilities(double ***summaries, double *test_row, int class_num, int *class_num_list, int row, int col)
{
	int i, j;
	double *probabilities = (double *)malloc(class_num * sizeof(double));
	for (i = 0; i < class_num; i++)
	{
		probabilities[i] = (double)class_num_list[i] / row;
	}
	for (i = 0; i < class_num; i++)
	{
		for (j = 0; j < col - 1; j++)
		{
			probabilities[i] *= calculate_probability(test_row[j], summaries[i][j][0], summaries[i][j][1]);
		}
	}
	return probabilities;
}
double predict(double ***summaries, double *test_row, int class_num, int *class_num_list, int row, int col)
{
	int i;
	double *probabilities = calculate_class_probabilities(summaries, test_row, class_num, class_num_list, row, col);
	double label = 0;
	double best_prob = probabilities[0];
	for (i = 1; i < class_num; i++)
	{
		if (probabilities[i] > best_prob)
		{
			label = i;
			best_prob = probabilities[i];
		}
	}
	return label;
}

void main()
{
	char filename[] = "iris.csv";
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
	int n_folds = 5;
	evaluate_algorithm(dataset, row, col, n_folds);
}
```

##### 8) compile.sh

```bash
gcc main.c read_csv.c normalize.c k_fold.c evaluate.c score.c test_prediction.c -o run -lm && ./run
```

**编译&运行：**

```bash
bash compile.sh
```

最终输出结果如下：

```c
Scores[0] = 96.666667%
Scores[1] = 93.333333%
Scores[2] = 96.666667%
Scores[3] = 100.000000%
Scores[4] = 93.333333%
mean_accuracy = 96.000000%
```

#### Python语言实战

本节同样假设您已经下载数据集，我们使用著名机器学习开源库sklearn高效实现**朴素贝叶斯算法**，以便您在实战中使用该算法：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':
    dataset = np.array(pd.read_csv("iris.csv", sep=',', header=None))
    k_Cross = KFold(n_splits=5, random_state=0, shuffle=True)
    index = 0
    score = np.array([])
    Scaler = MinMaxScaler()
    data,label = dataset[:,:-1],dataset[:,-1]
    data = Scaler.fit_transform(data)
    for train_index, test_index in k_Cross.split(dataset):
        train_data, train_label = data[train_index, :], label[train_index]
        test_data, test_label = data[test_index, :], label[test_index]
        model = GaussianNB()
        model.fit(train_data, train_label)
        pred = model.predict(test_data)
        acc = accuracy_score(test_label, pred)
        score = np.append(score,acc)
        print('score[{}] = {}%'.format(index,acc))
        index+=1
    print('mean_accuracy = {}%'.format(np.mean(score)))
```

输出结果如下：

```python
score[0] = 0.9666666666666667%
score[1] = 0.9%
score[2] = 0.9666666666666667%
score[3] = 1.0%
score[4] = 0.9333333333333333%
mean_accuracy = 0.9533333333333334%
```

