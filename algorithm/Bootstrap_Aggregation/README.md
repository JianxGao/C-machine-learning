## 3.11 Bootstrap Aggregation

> 我们前面已经介绍了决策树算法的C语言实现。决策树是一种简单而强大的预测模型，可是决策树可能面临方差过大的问题，即输入数据的细微偏差会导致预测结果的较大不同，这使得训练出来的结果对于特定的训练数据而言并不具有良好的泛化能力。本节将介绍的算法可以使得决策树拥有更高的鲁棒性，进而在预测中取得更好的表现。这种算法称为bootstrap aggregation（引导聚合），简称bagging（袋装）。它的主要思想是，在原始数据集上，通过**有放回抽样**的方法，重新选择出S个新数据集来分别训练S个分类器的集成技术。也就是说，这些模型训练的数据中允许存在重复的数据。在学习完本节的内容后，您将理解：如何从数据集中创建一个自助样本（bootstrap sample），如何运用这样的自助模型（bootstrapped models）来做预测，以及如何在你自己的预测问题上运用这些技巧。

### 3.11.1 算法介绍

引导（bootstrap）指的是对原始数据集做某种替换形成的样本。这意味着根据现有数据集中随机抽取的样本创建一个新的数据集，这种抽取是有放回的。当可用的数据集比较有限时，这是一种有用的方法。Bootstrap Aggregation（下文简称bagging），其基本思想是给定一个弱学习算法和一个训练集，由于单个弱学习算法准确率不高，所以我们将该学习算法使用多次，得出预测函数序列进行投票，依据得票数的多少来决定预测结果。在本节中，我们将重点介绍以决策树（相关基本知识可参阅本书 **3.5 Classification and Regression Trees**）为基分类器的bagging算法的C语言实现，并以Sonar数据集为例进行演示。

单个决策树虽然原理简单，但其方差较大，这意味着数据的更改越大，算法的性能变化也越大。高方差机器学习算法的性能就可以通过训练许多模型（如许多棵决策树），并取其预测的平均值来提高。这样的结果通常比单个模型要好。

除了提高性能外，Bagging的另一个优势是多个模型的叠加不会产生过拟合，因此我们可以放心地继续添加基分类器，直至满足性能要求。

### 3.11.2 算法讲解

#### 引导重采样

Bagging算法的重要内容就是对原始数据集进行有放回重采样，形成一系列子样本。显然，每个样本的抽取可以由随机数实现，子数据集中样本数占原数据集中样本数的比例可预先给定，由此可决定抽取样本的次数。

```{c}
double subsample(double **dataset, double ratio)
{
    int n_sample = (int)(row * ratio + 0.5);
    double **sample=0;
    sample = (double **)malloc(n_sample*sizeof(int *));
    for (int i = 0; i < n_sample; i++){
        sample[i] = (double *)malloc(col*sizeof(double));
    }
    int i,j,Index;
    for(i=0;i<n_sample;i++)
    {
        Index = rand() % row;
        for(j=0;j<col;j++)
        {
            sample[i][j] = dataset[Index][j];
        }
    }
    return **sample;
}
```

下面我们通过一个具体的例子来体验引导重采样的威力。

首先我们随机生成一个较小的数据集（20个介于0～10之间的整数）。

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main() {
    int a[20];
    srand((unsigned int)time(NULL));
    for(int i=0; i<20; i++) {
        a[i] = rand()%10;
        printf("%d  ",a[i]);
    }
    return 0;
}
```

输出：（此处输出仅作数据集生成函数效用的参考，以下演示用到的具体数据集在再次生成的时候必然不同于此）

```c
// Output:
8  9  2  7  5  1  5  2  5  7  5  2  1  9  9  6  2  4  6  3
```

下面我们比较一下不同的采样率得到的子数据集，它们的均值：

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    int a[20];
    srand((unsigned int)time(NULL));
    for(int i=0; i<20; i++) {
        a[i] = rand()%10;
        printf("%d\n",a[i]);
    }

    double ratio=0.1;
    double mean=0;
    int Index;
    int n_sample = (int)(20 * ratio + 0.5);

    // 计算子采样的均值
    for (int i = 0; i < n_sample; i++){
        Index = rand() % 20;
        mean+=a[Index];
    }
    mean/=n_sample;

    // 计算原数据的均值
    double sum=0;
    double Tmean=0;
    for(int i=0; i<20; i++) {
        sum+=a[i];
    }
    Tmean=sum/20;

    printf("\n子采样的均值为%.3f",mean);
    printf("\n数据集的均值为%.3f",Tmean);
    return 0;
}
```

当采样率为0.1时，可以发现子样本与原样本差别较大：

```c
// Output:
0  8  6  9  3  3  8  3  6  1  2  9  9  9  4  3  4  8  0  2  
子采样的均值为7.000
数据集的均值为4.850
```

随着采样率的提高，子样本越来越能够反映真实值。这与我们的常识也是相符的。

```c
// ratio=0.2
1  8  4  8  4  7  1  7  8  4  8  6  3  3  9  0  2  8  9  1  
子采样的均值为7.000
数据集的均值为5.050

// ratio=0.3
9  9  4  5  8  6  3  8  9  6  0  0  5  4  3  6  9  4  8  5  
子采样的均值为4.667
数据集的均值为5.550

// ratio=0.4
9  5  9  2  1  8  6  0  8  3  1  0  6  7  6  9  8  4  2  8  
子采样的均值为4.625
数据集的均值为5.100

// ratio=0.6
2  8  2  6  6  7  5  2  5  5  4  0  6  8  5  1  1  8  5  2  
子采样的均值为4.667
数据集的均值为4.400

// ratio=0.8
0  9  9  3  0  6  7  3  6  7  3  4  8  9  9  7  8  4  4  6  
子采样的均值为5.250
数据集的均值为5.600

// ratio=1.0
6  7  5  4  0  4  2  3  6  7  4  1  9  5  8  1  0  6  5  8  
子采样的均值为4.650
数据集的均值为4.550
```

#### 构建决策树

该步骤原理与代码可以参考**3.5 Classification and Regression Trees**。

### 3.11.1 算法代码

我们现在知道了如何实现**Bootstrap Aggregation**算法，那么我们把它应用到[声纳数据集 sonar.csv](https://aistudio.baidu.com/aistudio/datasetdetail/105756/0)

我们给出链接：https://aistudio.baidu.com/aistudio/datasetdetail/105756/0

#### C语言细节讲解

本节假设您已下载数据集 `sonar.csv`，并且它在当前工作目录中可用。下面我们给出一个完整实例，使用C语言详细讲解每一处细节。我们给出每一个.c文件的所有代码：

##### 1) read_csv.c

该步骤代码与前面CART部分相似，不再重复给出。

##### 2) k_fold.c

该步骤代码与前面CART部分相似，不再重复给出。

##### 3) BA.c

```c
#include "BA.h"

// 切分函数，根据切分点将数据分为左右两组
struct dataset *test_split(int index, double value, int row, int col, double **data)
{
	// 将切分结果作为结构体返回
	struct dataset *split = (struct dataset *)malloc(sizeof(struct dataset));
	int count1 = 0, count2 = 0;
	double ***groups = (double ***)malloc(2 * sizeof(double **));
	for (int i = 0; i < 2; i++)
	{
		groups[i] = (double **)malloc(row * sizeof(double *));
		for (int j = 0; j < row; j++)
		{
			groups[i][j] = (double *)malloc(col * sizeof(double));
		}
	}
	for (int i = 0; i < row; i++)
	{
		if (data[i][index] < value)
		{
			groups[0][count1] = data[i];
			count1++;
		}
		else
		{
			groups[1][count2] = data[i];
			count2++;
		}
	}
	split->splitdata = groups;
	split->row1 = count1;
	split->row2 = count2;
	return split;
}

// 计算Gini系数
double gini_index(int index, double value, int row, int col, double **dataset, double *class, int classnum)
{
	double *numcount1 = (double *)malloc(classnum * sizeof(double));
	double *numcount2 = (double *)malloc(classnum * sizeof(double));
	for (int i = 0; i < classnum; i++)
		numcount1[i] = numcount2[i] = 0;

	double count1 = 0, count2 = 0;
	double gini1, gini2, gini;
	gini1 = gini2 = gini = 0;
	// 计算每一类的个数
	for (int i = 0; i < row; i++)
	{
		if (dataset[i][index] < value)
		{
			count1++;
			for (int j = 0; j < classnum; j++)
				if (dataset[i][col - 1] == class[j])
					numcount1[j] += 1;
		}
		else
		{
			count2++;
			for (int j = 0; j < classnum; j++)
				if (dataset[i][col - 1] == class[j])
					numcount2[j]++;
		}
	}
	// 判断分母是否为0，防止运算错误
	if (count1 == 0)
	{
		gini1 = 1;
		for (int i = 0; i < classnum; i++)
			gini2 += (numcount2[i] / count2) * (numcount2[i] / count2);
	}
	else if (count2 == 0)
	{
		gini2 = 1;
		for (int i = 0; i < classnum; i++)
			gini1 += (numcount1[i] / count1) * (numcount1[i] / count1);
	}
	else
	{
		for (int i = 0; i < classnum; i++)
		{
			gini1 += (numcount1[i] / count1) * (numcount1[i] / count1);
			gini2 += (numcount2[i] / count2) * (numcount2[i] / count2);
		}
	}
	// 计算Gini系数
	gini1 = 1 - gini1;
	gini2 = 1 - gini2;
	gini = (count1 / row) * gini1 + (count2 / row) * gini2;
	free(numcount1);
	free(numcount2);
	numcount1 = numcount2 = NULL;
	return gini;
}

// 选取数据的最优切分点
struct treeBranch *get_split(int row, int col, double **dataset, double *class, int classnum, int n_features)
{
	struct treeBranch *tree = (struct treeBranch *)malloc(sizeof(struct treeBranch));
	int *featurelist = (int *)malloc(n_features * sizeof(int));
	int count = 0, flag = 0, temp;
	int b_index = 999;
	double b_score = 999, b_value = 999, score;
	// 随机选取n_features个特征
	while (count < n_features)
	{
		featurelist[count] = count;
		count++;
	}
	// 计算所有切分点Gini系数，选出Gini系数最小的切分点
	for (int i = 0; i < n_features; i++)
	{
		for (int j = 0; j < row; j++)
		{
			double value = dataset[j][featurelist[i]];
			score = gini_index(featurelist[i], value, row, col, dataset, class, classnum);
			if (score < b_score)
			{
				b_score = score;
				b_value = value;
				b_index = featurelist[i];
			}
		}
	}
	tree->index = b_index;
	tree->value = b_value;
	tree->flag = 0;
	return tree;
}

// 计算叶节点结果
double to_terminal(int row, int col, double **data, double *class, int classnum)
{
	int *num = (int *)malloc(classnum * sizeof(classnum));
	double maxnum = 0;
	int flag = 0;
	// 计算所有样本中结果最多的一类
	for (int i = 0; i < classnum; i++)
		num[i] = 0;
	for (int i = 0; i < row; i++)
		for (int j = 0; j < classnum; j++)
			if (data[i][col - 1] == class[j])
				num[j]++;
	for (int j = 0; j < classnum; j++)
	{
		if (num[j] > flag)
		{
			flag = num[j];
			maxnum = class[j];
		}
	}
	free(num);
	num = NULL;
	return maxnum;
}

// 创建子树或生成叶节点
void split(struct treeBranch *tree, int row, int col, double **data, double *class, int classnum, int depth, int min_size, int max_depth, int n_features)
{
	// 判断是否已经达到最大层数
	if (depth >= max_depth)
	{
		tree->flag = 1;
		tree->output = to_terminal(row, col, data, class, classnum);
		return;
	}
	struct dataset *childdata = test_split(tree->index, tree->value, row, col, data);
	// 判断样本是否已被分为一边
	if (childdata->row1 == 0 || childdata->row2 == 0)
	{
		tree->flag = 1;
		tree->output = to_terminal(row, col, data, class, classnum);
		return;
	}
	// 左子树，判断样本是否达到最小样本数，如不是则继续迭代
	if (childdata->row1 <= min_size)
	{
		struct treeBranch *leftchild = (struct treeBranch *)malloc(sizeof(struct treeBranch));
		leftchild->flag = 1;
		leftchild->output = to_terminal(childdata->row1, col, childdata->splitdata[0], class, classnum);
		tree->leftBranch = leftchild;
	}
	else
	{
		struct treeBranch *leftchild = get_split(childdata->row1, col, childdata->splitdata[0], class, classnum, n_features);
		tree->leftBranch = leftchild;
		split(leftchild, childdata->row1, col, childdata->splitdata[0], class, classnum, depth + 1, min_size, max_depth, n_features);
	}
	// 右子树，判断样本是否达到最小样本数，如不是则继续迭代
	if (childdata->row2 <= min_size)
	{
		struct treeBranch *rightchild = (struct treeBranch *)malloc(sizeof(struct treeBranch));
		rightchild->flag = 1;
		rightchild->output = to_terminal(childdata->row2, col, childdata->splitdata[1], class, classnum);
		tree->rightBranch = rightchild;
	}
	else
	{
		struct treeBranch *rightchild = get_split(childdata->row2, col, childdata->splitdata[1], class, classnum, n_features);
		tree->rightBranch = rightchild;
		split(rightchild, childdata->row2, col, childdata->splitdata[1], class, classnum, depth + 1, min_size, max_depth, n_features);
	}
	free(childdata->splitdata);
	childdata->splitdata = NULL;
	free(childdata);
	childdata = NULL;
	return;
}

// 生成决策树
struct treeBranch *build_tree(int row, int col, double **data, int min_size, int max_depth, int n_features)
{
	int count1 = 0, flag1 = 0;
	// 判断结果一共有多少类别，此处classes[20]仅仅是取一个较大的数20，默认类别不可能超过20类
	double classes[20];
	for (int i = 0; i < row; i++)
	{
		if (count1 == 0)
		{
			classes[0] = data[i][col - 1];
			count1++;
		}
		else
		{
			flag1 = 0;
			for (int j = 0; j < count1; j++)
				if (classes[j] == data[i][col - 1])
					flag1 = 1;
			if (flag1 == 0)
			{
				classes[count1] = data[i][col - 1];
				count1++;
			}
		}
	}
	// 生成切分点
	struct treeBranch *result = get_split(row, col, data, classes, count1, n_features);
	// 进入迭代，不断生成子树
	split(result, row, col, data, classes, count1, 1, min_size, max_depth, n_features);
	return result;
}

// 随机森林算法，将森林结果保存为结构体数组，并返回结构体二重指针
struct treeBranch **random_forest(int row, int col, double **data, int min_size, int max_depth, int n_features, int n_trees, double sample_size)
{
	struct treeBranch **forest = (struct treeBranch **)malloc(n_trees * sizeof(struct treeBranch *));
	int samplenum = (int)(row * sample_size);
	int temp;
	// 生成随机训练集
	double **subsample = (double **)malloc(samplenum * sizeof(double *));
	for (int i = 0; i < samplenum; i++)
	{
		subsample[i] = (double *)malloc(col * sizeof(double));
	}
	// 生成所有决策树
	for (int j = 0; j < n_trees; j++)
	{
		for (int i = 0; i < samplenum; i++)
		{
			temp = rand() % row;
			subsample[i] = data[temp];
		}
		struct treeBranch *tree = build_tree(samplenum, col, subsample, min_size, max_depth, n_features);
		forest[j] = tree;
	}
	return forest;
}

// 决策树预测
double treepredict(double *test, struct treeBranch *tree)
{
	double output;
	// 判断是否达到叶节点，flag=1时为叶节点，flag=0时则继续判断
	if (tree->flag == 1)
	{
		output = tree->output;
		return output;
	}
	else
	{
		if (test[tree->index] < tree->value)
		{
			output = treepredict(test, tree->leftBranch);
			return output;
		}
		else
		{
			output = treepredict(test, tree->rightBranch);
			return output;
		}
	}
}

// 随机森林bagging预测
double predict(double *test, struct treeBranch **forest, int n_trees)
{
	double output;
	double *forest_result = (double *)malloc(n_trees * sizeof(double));
	double *classes = (double *)malloc(n_trees * sizeof(double));
	int *num = (int *)malloc(n_trees * sizeof(int));
	for (int i = 0; i < n_trees; i++)
		num[i] = 0;
	int count = 0, flag, temp = 0;
	// 将每棵树的判断结果保存
	for (int i = 0; i < n_trees; i++)
		forest_result[i] = treepredict(test, forest[i]);
	// bagging选出最后结果
	for (int i = 0; i < n_trees; i++)
	{
		flag = 0;
		for (int j = 0; j < count; j++)
		{
			if (forest_result[i] == classes[j])
			{
				flag = 1;
				num[j]++;
			}
		}
		if (flag == 0)
		{
			classes[count] = forest_result[i];
			num[count]++;
			count++;
		}
	}
	for (int i = 0; i < count; i++)
	{
		if (num[i] > temp)
		{
			temp = num[i];
			output = classes[i];
		}
	}
	return output;
}

// 从样本集中进行有放回子采样得到子样本集
double **subsample(double **dataset, double ratio)
{
	int n_sample = (int)(row * ratio + 0.5);
	double **sample = 0;
	sample = (double **)malloc(n_sample * sizeof(int *));
	for (int i = 0; i < n_sample; i++)
	{
		sample[i] = (double *)malloc(col * sizeof(double));
	}
	int i, j, Index;
	for (i = 0; i < n_sample; i++)
	{
		Index = rand() % row;
		for (j = 0; j < col; j++)
		{
			sample[i][j] = dataset[Index][j];
		}
	}
	return sample;
}

// 用每个基模型进行预测并投票决定最终预测结果
double bagging_predict(struct treeBranch **trees, double *row, int n_trees)
{
	int i;
	int n_classes = 1000;
	int class_count[1000] = {0};
	double *predictions = 0;
	predictions = (double *)malloc(n_trees * sizeof(double));
	double counts = 0;
	double result = 0;
	for (i = 0; i < n_trees; i++)
	{
		predictions[i] = predict(row, &trees[i], n_trees);
		class_count[(int)predictions[i]] += 1;
	}
	for (i = 0; i < n_classes; i++)
	{
		if (class_count[i] > counts)
		{
			counts = class_count[i];
			result = i;
		}
	}
	return result;
}

// Bagging 算法实现
double bagging(double **train, double **test, int max_depth, int min_size, double sample_size, int n_trees, int n_features)
{
	int i, j;
	int n_classes = 1000;
	int class_count[1000] = {0};
	double **predictions = 0;
	predictions = (double **)malloc(n_trees * sizeof(int));
	for (int i = 0; i < n_trees; i++)
	{
		predictions[i] = (double *)malloc(sizeof(test) / sizeof(test[0]) * sizeof(double));
	}
	double counts = 0;
	double result = 0;
	double *results = 0;
	results = (double *)malloc(sizeof(test) / sizeof(test[0]) * sizeof(double *));
	for (i = 0; i < n_trees; i++)
	{
		double **sample = subsample(train, sample_size); // sample_size表示ratio
		struct treeBranch *tree = build_tree(row, col, sample, min_size, max_depth, n_features);
		for (j = 0; j < sizeof(test) / sizeof(test[0]); j++)
		{
			predictions[i][j] = predict(test[j], &tree, n_features);
		}
	}
	for (j = 0; j < sizeof(test) / sizeof(test[0]); j++)
	{
		for (i = 0; i < n_trees; i++)
		{
			class_count[(int)predictions[i][j]] += 1;
		}
		for (i = 0; i < n_classes; i++)
		{
			if (class_count[i] > counts)
			{
				counts = class_count[i];
				result = i;
			}
		}
		results[j] = result;
	}
	return *results;
}
```

##### 4) BA.h

```c
#ifndef BA
#define BA

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// 读取csv数据，全局变量
double **dataset;
int row, col;

// 树的结构体，flag判断是否为叶节点，index和value为切分点，Brance为对应子树
struct treeBranch
{
	int flag;
	int index;
	double value;
	double output;
	struct treeBranch *leftBranch;
	struct treeBranch *rightBranch;
};

// 切分数据，splitdata为切分成左右两组的三维数据，row1为左端数据行数，row2为右端
struct dataset
{
	int row1;
	int row2;
	double ***splitdata;
};

int get_row(char *filename);
int get_col(char *filename);
void get_two_dimension(char *line, double **dataset, char *filename);
double ***cross_validation_split(double **dataset, int row, int n_folds, int fold_size);
double *get_test_prediction(double **train, double **test, int column, int min_size, int max_depth, int n_features, int n_trees, double sample_size, int fold_size, int train_size);
struct dataset *test_split(int index, double value, int row, int col, double **data);
double gini_index(int index, double value, int row, int col, double **dataset, double *class, int classnum);
struct treeBranch *get_split(int row, int col, double **dataset, double *class, int classnum, int n_features);
double to_terminal(int row, int col, double **data, double *class, int classnum);
void split(struct treeBranch *tree, int row, int col, double **data, double *class, int classnum, int depth, int min_size, int max_depth, int n_features);
struct treeBranch *build_tree(int row, int col, double **data, int min_size, int max_depth, int n_features);
struct treeBranch **random_forest(int row, int col, double **data, int min_size, int max_depth, int n_features, int n_trees, double sample_size);
double treepredict(double *test, struct treeBranch *tree);
double predict(double *test, struct treeBranch **tree, int n_trees);
double accuracy_metric(double *actual, double *predicted, int fold_size);
double *evaluate_algorithm(double **dataset, int column, int n_folds, int fold_size, int min_size, int max_depth, int n_features, int n_trees, double sample_size);

#endif
```

##### 5) score.c

该步骤代码与前面CART部分相似，不再重复给出。

##### 6) test_prediction.c

```c
#include "BA.h"

double *get_test_prediction(double **train, double **test, int column, int min_size, int max_depth, int n_features, int n_trees, double sample_size, int fold_size, int train_size)
{
	double *predictions = (double *)malloc(fold_size * sizeof(double)); //预测集的行数就是数组prediction的长度
	struct treeBranch **forest = random_forest(train_size, column, train, min_size, max_depth, n_features, n_trees, sample_size);
	for (int i = 0; i < fold_size; i++)
	{
		predictions[i] = predict(test[i], forest, n_trees);
	}
	return predictions; //返回对test的预测数组
}
```

##### 7) evaluate.c

```c
#include "BA.h"

double *evaluate_algorithm(double **dataset, int column, int n_folds, int fold_size, int min_size, int max_depth, int n_features, int n_trees, double sample_size)
{
	double ***split = cross_validation_split(dataset, row, n_folds, fold_size);
	int i, j, k, l;
	int test_size = fold_size;
	int train_size = fold_size * (n_folds - 1); //train_size个一维数组
	double *score = (double *)malloc(n_folds * sizeof(double));
	for (i = 0; i < n_folds; i++)
	{ //因为要遍历删除，所以拷贝一份split
		double ***split_copy = (double ***)malloc(n_folds * sizeof(double **));
		for (j = 0; j < n_folds; j++)
		{
			split_copy[j] = (double **)malloc(fold_size * sizeof(double *));
			for (k = 0; k < fold_size; k++)
			{
				split_copy[j][k] = (double *)malloc(column * sizeof(double));
			}
		}
		for (j = 0; j < n_folds; j++)
		{
			for (k = 0; k < fold_size; k++)
			{
				for (l = 0; l < column; l++)
				{
					split_copy[j][k][l] = split[j][k][l];
				}
			}
		}
		double **test_set = (double **)malloc(test_size * sizeof(double *));
		for (j = 0; j < test_size; j++)
		{ //对test_size中的每一行
			test_set[j] = (double *)malloc(column * sizeof(double));
			for (k = 0; k < column; k++)
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
				train_set[k * fold_size + l] = (double *)malloc(column * sizeof(double));
				train_set[k * fold_size + l] = split_copy[k][l];
			}
		}
		double *predicted = (double *)malloc(test_size * sizeof(double)); //predicted有test_size个
		predicted = get_test_prediction(train_set, test_set, column, min_size, max_depth, n_features, n_trees, sample_size, fold_size, train_size);
		double *actual = (double *)malloc(test_size * sizeof(double));
		for (l = 0; l < test_size; l++)
		{
			actual[l] = test_set[l][column - 1];
		}
		double accuracy = accuracy_metric(actual, predicted, test_size);
		score[i] = accuracy;
		printf("score[%d]=%f%%\n", i, score[i]);
		free(split_copy);
	}
	double total = 0.0;
	for (l = 0; l < n_folds; l++)
	{
		total += score[l];
	}
	printf("mean_accuracy=%f%%\n", total / n_folds);
	return score;
}
```

##### 8) main.c

```c
#include "BA.h"

int main()
{
	char filename[] = "sonar.csv";
	char line[1024];
	row = get_row(filename);
	col = get_col(filename);
	int n_features = col - 1;
	dataset = (double **)malloc(row * sizeof(int *));
	for (int i = 0; i < row; ++i)
	{
		dataset[i] = (double *)malloc(col * sizeof(double));
	} //动态申请二维数组
	get_two_dimension(line, dataset, filename);

	// 输入模型参数，包括每个叶子最小样本数、最大层数、树木个数
	int min_size = 2, max_depth = 10, n_trees = 20;
	double sample_size = 1;
	int n_folds = 8;
	int fold_size = (int)(row / n_folds);

	// Bagging算法，返回交叉验证正确率
	double *score = evaluate_algorithm(dataset, col, n_folds, fold_size, min_size, max_depth, n_features, n_trees, sample_size);
}
```

##### 9) compile.sh

```bash
gcc main.c read_csv.c BA.c k_fold.c evaluate.c score.c test_prediction.c -o run -lm && ./run
```

**编译&运行：**

```bash
bash compile.sh
```

运算后得到的结果如下：

```c
score[0] = 84.615385%
score[1] = 76.923077%
score[2] = 92.307692%
score[3] = 65.384615%
score[4] = 96.153846%
score[5] = 80.769231%
score[6] = 76.923077%
score[7] = 80.769231%
mean_accuracy = 81.730769%
```

#### Python语言实战

本节同样假设您已经下载数据集，我们使用著名机器学习开源库sklearn高效实现**Bootstrap Aggregation算法**，以便您在实战中使用该算法：

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier


if __name__ == '__main__':
    dataset = np.array(pd.read_csv("sonar.csv", sep=',', header=None))
    k_Cross = KFold(n_splits=8, random_state=0, shuffle=True)
    index = 0
    score = np.array([])
    data,label = dataset[:,:-1],dataset[:,-1]
    for train_index, test_index in k_Cross.split(dataset):
        train_data, train_label = data[train_index, :], label[train_index]
        test_data, test_label = data[test_index, :], label[test_index]
        tree = DecisionTreeClassifier()
        model = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True, random_state=1)
        model.fit(train_data, train_label)
        pred = model.predict(test_data)
        acc = accuracy_score(test_label, pred)
        score = np.append(score,acc)
        print('score[{}] = {}%'.format(index,acc)) * 100
        index+=1
    print('mean_accuracy = {}%'.format(np.mean(score)))
```

输出结果如下：

```python
score[0] = 76.92307692307693%
score[1] = 84.61538461538461%
score[2] = 73.07692307692307%
score[3] = 69.23076923076923%
score[4] = 80.76923076923077%
score[5] = 80.76923076923077%
score[6] = 84.61538461538461%
score[7] = 80.76923076923077%
mean_accuracy = 78.84615384615384%
```

