## 3.9 Support Vector Machine

> 支持向量机（Support Vector Machine, SVM）是一类按监督学习方式对数据进行二元分类的广义线性分类器，其决策边界是对学习样本求解的最大边距超平面。SVM算法最早是由 Vladimir N. Vapnik 和 Alexey Ya. Chervonenkis 在1963年提出；目前的版本(soft margin)是由Corinna Cortes 和 Vapnik在1993年提出，并在1995年发表；深度学习（2012）出现之前，SVM被认为机器学习中近十几年来最成功，表现最好的算法。

### 3.9.1 算法简介

支持向量机（SVM）是用于分类与回归分析中分析数据的监督式学习模型与相关的学习算法。给定一组数据集，每个训练数据被标记为属于两个类别中的一个或另一个，SVM训练算法创建一个根据训练集的模型，使该模型成为非概率二元线性分类器。除了进行线性分类之外，SVM还可以使用所谓的核技巧有效地进行非线性分类，将其输入隐式映射到高维特征空间中。

本书中，我们将主要介绍数据集**线性可分**情况下的SVM算法。

### 3.9.2 算法讲解

#### 超平面

SVM算法的目标是找到一个**超平面** $\omega X+b=0$，以此超平面为边界，可以将数据集分为两类，如下图所示：

![image-20210827134231474](.\image.png)

因此，我们需要先找到各个分类的样本点离这个超平面最近的点，使得这个点到超平面的距离最大化，最近的点就是在虚线上的点，这个点也被称之为**支持向量**。于是，我们认为$\omega  X-b>1$为蓝色点，$\omega  X-b>-1$为绿色点，于是完成二分类。

#### 线性可分

在数据集式线性可分的情况下，数据集应分布于超平面的两侧。那么，使得支持向量到超平面的距离最大化，可以使用如下等价的两个优化式表示：
$$
\begin{array}{llrl}
\max _{\boldsymbol{w}, b} & \frac{2}{\|\boldsymbol{w}\|} & \Longleftrightarrow \min _{\boldsymbol{w}, b} & \frac{1}{2}\|\boldsymbol{w}\|^{2} \\
\text { s.t. } & y_{i}\left(\boldsymbol{w}^{\top} \boldsymbol{X}_{i}+b\right) \geq 1 & \text { s.t. } & y_{i}\left(\boldsymbol{w}^{\top} \boldsymbol{X}_{i}+b\right) \geq 1
\end{array}
$$

其中，$\omega=\sum_{i=1}^{n} \alpha_{i}t_{i} x_{i}$，$x_i$表示数据集特正，$t_i$表示label。

我们可以通过复杂的数学推导（此处省略），求解出$\omega$和$b$的值，进而确定我们的算法参数。为此，我们介绍简化版的SMO算法，该算法可以求解参数$\alpha$和$b$，进而我们可以确定$\omega$。

#### SMO求解

我们首先给出算法的输入输出：

<img src=".\image2.png" alt="image-20210827145333856" style="zoom:65%;" />

进而，我们给出算法的伪代码：

<img src="./image3.png" alt="image-20210827145429600" style="zoom:90%;" />

其中，上述为代码中使用的公式如下所示：
$$
\begin{gathered}
f(x)=\omega x^{T}+b=\sum_{i=1}^{n} \alpha_{i}^{o l d} t_{i} x_{i} x^{T}+b &(45)\\
t_{i} \neq t_{j} \Longrightarrow L=\max \left(0, \alpha_{j}^{o l d}-\alpha_{i}^{\text {old }}\right), \quad H=\min \left(C, C+\alpha_{j}^{\text {old }}-\alpha_{i}^{\text {old }}\right) & (46)\\
t_{i}=t_{j} \Longrightarrow L=\max \left(0, \alpha_{j} v+\alpha_{i}^{\text {old }}-C\right), \quad H=\min \left(C, \alpha_{j}^{\text {old }}+\alpha_{i}^{\text {old }}\right) & (47)\\
E_{k}=f\left(x_{k}\right)-t_{k}, \quad \eta=2 x_{i} x_{j}^{T}-x_{i} x_{i}^{T}-x_{j} x_{j}^{T} & (48)\\
\alpha_{j}^{\text {raw }}=\alpha_{j}^{\text {old }}-\frac{t_{j}\left(E_{i}-E_{j}\right)}{\eta}&(49)\\
\alpha_{j}^{\text {new }}= \begin{cases}H & \text { if } \alpha_{j}^{\text {raw }}>H  \\
\alpha_{j}^{\text {raw }} & \text { if } L \leq \alpha_{j}^{\text {raw }} \leq H  \\
L & \text { if } \alpha_{j}^{\text {raw }}<L .\end{cases} &(50)\\
\qquad \alpha_{i}^{\text {new }}=\alpha_{i}^{\text {old }}+t_{i} t_{j}\left(\alpha_{j}^{\text {old }}-\alpha_{j}^{\text {new }}\right) &(51) \\
b_{1}=b-E_{i}-t_{i}\left(\alpha_{i}^{\text {new }}-\alpha_{i}^{\text {old }}\right) x_{i} x_{i}^{T}-t_{j}\left(\alpha_{j}^{\text {new }}-\alpha_{j}^{\text {old }}\right) x_{i} x_{j}^{T} &(52) \\
b_{2}=b-E_{j}-t_{i}\left(\alpha_{i}^{\text {new }}-\alpha_{i}^{\text {old }}\right) x_{i} x_{j}^{T}-t_{j}\left(\alpha_{j}^{\text {new }}-\alpha_{j}^{\text {old }}\right) x_{j} x_{j}^{T} &(53) \\
b^{\text {new }}= \begin{cases}b_{1} & \text { if } 0<\alpha_{i}^{\text {new }}<C \\
b_{2} & \text { if } 0<\alpha_{j}^{\text {new }}<C \\
\left(b_{1}+b_{2}\right) / 2 & \text { otherwise. }\end{cases}&(54)
\end{gathered}
$$
我们给出该函数`Svm_Smo`的代码：

```c
void Svm_Smo(double *b, double *alpha, int m_passes, double **train_data, double *label, double tol, double C, double change_limit, int row, int col)
{
    srand((unsigned)time(NULL));
    int p_num = 0;
    while (p_num < m_passes)
    {
        int num_chaged_alpha = 0;
        for (int i = 0; i < row; i++)
        {
            double error_i = calculate_error(*b, label, train_data, alpha, row, col, i);
            if (((label[i] * error_i < (-tol)) && (alpha[i] < C)) || ((label[i] * error_i > tol) && (alpha[i] > 0)))
            {
                int j = rand() % row;
                while (j == i)
                {
                    j = rand() % row;
                }
                double error_j = calculate_error(*b, label, train_data, alpha, row, col, j);
                // save old alpha i, j
                double alpha_old_i = alpha[i];
                double alpha_old_j = alpha[j];
                // compute Land H
                double L = 0, H = C;
                if (label[i] != label[j])
                {
                    double L = 0 > (alpha[j] - alpha[i]) ? 0 : (alpha[j] - alpha[i]);
                    double H = C < (C + alpha[j] - alpha[i]) ? C : (C + alpha[j] - alpha[i]);
                }
                else
                {
                    double L = 0 > (alpha[j] + alpha[i] - C) ? 0 : (alpha[j] + alpha[i] - C);
                    double H = C < (alpha[j] + alpha[i]) ? C : (alpha[j] + alpha[i]);
                }
                if (L == H)
                {
                    continue;
                }
                // compute eta, in order to be convenient to judge
                double eta = 2 * array_dot(train_data[i], train_data[j], col - 1) -
                    array_dot(train_data[i], train_data[i], col - 1) -
                    array_dot(train_data[j], train_data[j], col - 1);
                if (eta >= 0)
                {
                    continue;
                }
                // computeand clip new value for alpha_raw_j
                alpha[j] -= (label[j] * (error_i - error_j) / eta);
                // compute alpha_new_j
                if (alpha[j] > H)
                {
                    alpha[j] = H;
                }
                else if (alpha[j] < L)
                {
                    alpha[j] = L;
                }
                // Check
                if (fabs(alpha[j] - alpha_old_j) < change_limit)
                {
                    continue;
                }
                // compute alpha_new_i
                alpha[i] += label[i] * label[j] * (alpha_old_j - alpha[j]);
                // compute b1, b2
                double b1 = *b - error_i - label[i] * (alpha[i] - alpha_old_i) * array_dot(train_data[i], train_data[i], col - 1) -
                    label[j] * (alpha[j] - alpha_old_j) * array_dot(train_data[i], train_data[j], col - 1);
                double b2 = *b - error_j - label[i] * (alpha[i] - alpha_old_i) * array_dot(train_data[i], train_data[j], col - 1) -
                    label[j] * (alpha[j] - alpha_old_j) * array_dot(train_data[j], train_data[j], col - 1);
                if ((0 < alpha[i]) && (alpha[i] < C))
                {
                    *b = b1;
                }
                else if ((0 < alpha[j]) && (alpha[j] < C))
                {
                    *b = b2;
                }
                else
                {
                    *b = (b1 + b2) / 2;
                    num_chaged_alpha += 1;
                }
            }
            else
            {
                continue;
            }
        }
        if (num_chaged_alpha == 0)
        {
            p_num += 1;
        }
        else
        {
            p_num = 0;
        }
    }
}
```

如此我们求得参数$\alpha$和$b$，再通过函数`get_weight`，即可求得$\omega$的值：

```c
double *get_weight(double *alpha, double *label, double **train_data, int row, int col)
{
    double *weight;
    weight = (double *)malloc((col - 1) * sizeof(double));
    for (int j = 0; j < col - 1; j++)
    {
        weight[j] = 0;
        for (int i = 0; i < row; i++)
        {
            weight[j] += alpha[i] * label[i] * train_data[i][j];
        }
    }
    return weight;
}
```

### 3.9.3 算法代码

我们现在知道了如何实现**支持向量机算法**，那么我们把它应用到[声纳数据集 sonar.csv](https://aistudio.baidu.com/aistudio/datasetdetail/105756/0)

我们给出链接：https://aistudio.baidu.com/aistudio/datasetdetail/105756/0

#### C语言细节讲解

本节假设您已下载数据集 `sonar.csv`，并且它在当前工作目录中可用。下面我们给出一个完整实例，使用C语言详细讲解每一处细节。我们给出每一个.c文件的所有代码：

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
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

double *get_label(double **dataset, int row, int col)
{
    double *label = (double *)malloc(row * sizeof(double));
    for (int i = 0; i < row; i++)
    {
        label[i] = dataset[i][col - 1];
    }
    return label;
}

double array_dot(double *row1, double *row2, int col)
{
    double res = 0;
    for (int i = 0; i < col; i++)
    {
        res += row1[i] * row2[i];
    }
    return res;
}

double calculate_error(double b, double *label, double **data, double *alpha, int row, int column, int index)
{
    double error = 0;
    double *dot_res;
    dot_res = (double *)malloc(row * sizeof(double));
    for (int i = 0; i < row; i++)
    {
        dot_res[i] = array_dot(data[i], data[index], column - 1);

        dot_res[i] = dot_res[i] * alpha[i] * label[i];
        error += dot_res[i];
    }

    error += b - label[index];

    return error;
}

void Svm_Smo(double *b, double *alpha, int m_passes, double **train_data, double *label, double tol, double C, double change_limit, int row, int col)
{
    srand((unsigned)time(NULL));
    int p_num = 0;
    while (p_num < m_passes)
    {
        int num_chaged_alpha = 0;
        for (int i = 0; i < row; i++)
        {
            double error_i = calculate_error(*b, label, train_data, alpha, row, col, i);
            if (((label[i] * error_i < (-tol)) && (alpha[i] < C)) || ((label[i] * error_i > tol) && (alpha[i] > 0)))
            {
                int j = rand() % row;
                while (j == i)
                {
                    j = rand() % row;
                }
                double error_j = calculate_error(*b, label, train_data, alpha, row, col, j);
                // save old alpha i, j
                double alpha_old_i = alpha[i];
                double alpha_old_j = alpha[j];
                // compute Land H
                double L = 0, H = C;
                if (label[i] != label[j])
                {
                    double L = 0 > (alpha[j] - alpha[i]) ? 0 : (alpha[j] - alpha[i]);
                    double H = C < (C + alpha[j] - alpha[i]) ? C : (C + alpha[j] - alpha[i]);
                }
                else
                {
                    double L = 0 > (alpha[j] + alpha[i] - C) ? 0 : (alpha[j] + alpha[i] - C);
                    double H = C < (alpha[j] + alpha[i]) ? C : (alpha[j] + alpha[i]);
                }
                if (L == H)
                {
                    continue;
                }
                // compute eta, in order to be convenient to judge
                double eta = 2 * array_dot(train_data[i], train_data[j], col - 1) -
                    array_dot(train_data[i], train_data[i], col - 1) -
                    array_dot(train_data[j], train_data[j], col - 1);
                if (eta >= 0)
                {
                    continue;
                }
                // computeand clip new value for alpha_raw_j
                alpha[j] -= (label[j] * (error_i - error_j) / eta);
                // compute alpha_new_j
                if (alpha[j] > H)
                {
                    alpha[j] = H;
                }
                else if (alpha[j] < L)
                {
                    alpha[j] = L;
                }
                // Check
                if (fabs(alpha[j] - alpha_old_j) < change_limit)
                {
                    continue;
                }
                // compute alpha_new_i
                alpha[i] += label[i] * label[j] * (alpha_old_j - alpha[j]);
                // compute b1, b2
                double b1 = *b - error_i - label[i] * (alpha[i] - alpha_old_i) * array_dot(train_data[i], train_data[i], col - 1) -
                    label[j] * (alpha[j] - alpha_old_j) * array_dot(train_data[i], train_data[j], col - 1);
                double b2 = *b - error_j - label[i] * (alpha[i] - alpha_old_i) * array_dot(train_data[i], train_data[j], col - 1) -
                    label[j] * (alpha[j] - alpha_old_j) * array_dot(train_data[j], train_data[j], col - 1);
                if ((0 < alpha[i]) && (alpha[i] < C))
                {
                    *b = b1;
                }
                else if ((0 < alpha[j]) && (alpha[j] < C))
                {
                    *b = b2;
                }
                else
                {
                    *b = (b1 + b2) / 2;
                    num_chaged_alpha += 1;
                }
            }
            else
            {
                continue;
            }
        }
        if (num_chaged_alpha == 0)
        {
            p_num += 1;
        }
        else
        {
            p_num = 0;
        }
    }
}

double *get_weight(double *alpha, double *label, double **train_data, int row, int col)
{
    double *weight;
    weight = (double *)malloc((col - 1) * sizeof(double));
    for (int j = 0; j < col - 1; j++)
    {
        weight[j] = 0;
        for (int i = 0; i < row; i++)
        {
            weight[j] += alpha[i] * label[i] * train_data[i][j];
        }
    }
    return weight;
}

double predict(double *w, double *test_data, double b, int col)
{
    return array_dot(test_data, w, col - 1) + b;
}

double *get_test_prediction(double **train, int train_size, double **test_data, int test_size, int m_passes, double tol, double C, double change_limit, int col)
{
    double b = 0;
    double *alpha = (double *)malloc(train_size * sizeof(double));
    for (int i = 0; i < train_size; i++)
    {
        alpha[i] = 0;
    }
    double *label = get_label(train, train_size, col);
    Svm_Smo(&b, alpha, m_passes, train, label, tol, C, change_limit, train_size, col);
    double *w = get_weight(alpha, label, train, train_size, col);
    double *predictions = (double *)malloc(test_size * sizeof(double));
    for (int i = 0; i < test_size; i++)
    {
        predictions[i] = predict(w, test_data[i], b, col);
        if (predictions[i] >= 0)
        {
            predictions[i] = 1;
        }
        else
        {
            predictions[i] = -1;
        }
    }
    return predictions;
}
```

##### 6) evaluate.c

```c
#include <stdlib.h>
#include <stdio.h>

extern double *get_test_prediction(double **train, int train_size, double **test_data, int test_size, int m_passes, double tol, double C, double change_limit, int col);
extern double accuracy_metric(double *actual, double *predicted, int fold_size);
extern double ***cross_validation_split(double **dataset, int row, int col, int n_folds, int fold_size);

void evaluate_algorithm(double **dataset, int row, int col, int n_folds, int m_passes, double tol, double C, double change_limit)
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
		predicted = get_test_prediction(train_set, train_size, test_set, test_size, m_passes, tol, C, change_limit, col);
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
#include <stdio.h>
#include <stdlib.h>

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern void normalize_dataset(double **dataset, int row, int col);
extern void evaluate_algorithm(double **dataset, int row, int col, int n_folds, int m_passes, double tol, double c, double change_limit);

void main()
{
	char filename[] = "sonar.csv";
	char line[1024];
	int row = get_row(filename);
	int col = get_col(filename);
	double **dataset;
	dataset = (double **)malloc(row * sizeof(double *));
	for (int i = 0; i < row; ++i)
	{
		dataset[i] = (double *)malloc(col * sizeof(double));
	}
	get_two_dimension(line, dataset, filename);
	normalize_dataset(dataset, row, col);
	int n_folds = 5;
	int max_passes = 1;
	double C = 0.05;
	double tolerance = 0.001;
	double change_limit = 0.001;
	evaluate_algorithm(dataset, row, col, n_folds, max_passes, tolerance, C, change_limit);
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
Scores[0] = 56.097561%
Scores[1] = 53.658537%
Scores[2] = 56.097561%
Scores[3] = 51.219512%
Scores[4] = 53.658537%
mean_accuracy = 54.146341%
```

#### Python语言实战

本节同样假设您已经下载数据集，我们使用著名机器学习开源库sklearn高效实现**支持向量机算法**，以便您在实战中使用该算法：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    dataset = np.array(pd.read_csv("sonar.csv", sep=',', header=None))
    k_Cross = KFold(n_splits=5, random_state=0, shuffle=True)
    index = 0
    score = np.array([])
    data,label = dataset[:,:-1],dataset[:,-1]
    for train_index, test_index in k_Cross.split(dataset):
        train_data, train_label = data[train_index, :], label[train_index]
        test_data, test_label = data[test_index, :], label[test_index]
        model = SVC()
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
score[0] = 0.7857142857142857%
score[1] = 0.6904761904761905%
score[2] = 0.7804878048780488%
score[3] = 0.8536585365853658%
score[4] = 0.8292682926829268%
mean_accuracy = 0.7879210220673635%
```

