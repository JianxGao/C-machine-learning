## 3.12 Random Forest

> 我们前面已经介绍了决策树和Bagging算法的C语言实现。随机森林是基于Bagging的一个扩展，它在Bagging对原始训练数据进行有放回抽样形成子数据集的基础上，对构建决策树的特征也进行随机的选择。不同的树之间可能长得非常不同，这种做法可以提高算法的性能。在本节中，您将了解Bagging和随机森林的区别，如何由高方差的决策树构建随机森林，如何将随机森林算法应用于实际的预测问题中。

### 3.12.1 算法介绍

Bagging算法虽然已在一定程度上提高了单棵决策树的性能，但它也有一些限制。它对每一棵构建的树，应用相同的贪婪算法，这将导致每棵树中都会选中相似的分割点，也就是说每一棵树将长得非常相似。相似的树做出的预测也是相似的，这样让它们做投票就达不到预想的效果。为了解决这一问题，我们可以通过限制每一棵树分割点评估的特征来强制决策树不同，这种算法就叫做随机森林。和bagging相同的是，都需要获取原始数据集的多个子样本，并用之分别训练不同的决策树。与bagging不同的是，在每个点上对数据进行拆分并将其添加到树中时，只能考虑固定的特征子集，也就是说对每一棵树所关注的特征，也进行随机选择。

一般来说，对于分类问题，每一棵树考虑的分割特征数量限制为输入特征数量的平方根。即

$$
num\_features\_for\_split=\sqrt {total\_input\_features}
$$

如此一来，树与树之间的差异会更大，从而导致预测结果多元化，组合预测的性能将比单棵决策树或者只进行套袋要好很多。


### 3.12.2 算法讲解

#### 2.1 构建决策树

与构建单纯决策树的不同之处，主要体现在寻找最佳分点的方式。需要随机选取n_features（预先设定）个特征值用于树的构建。

```{c}
struct treeBranch *get_split(int row, int col, double **dataset, double *class, int classnum, int n_features)
{
    struct treeBranch *tree=(struct treeBranch *)malloc(sizeof(struct treeBranch));
    int *featurelist=(int *)malloc(n_features * sizeof(int));
    int count=0,flag=0,temp;
    int b_index=999;
    double b_score = 999, b_value = 999,score;
    // 随机选取n_features个特征
    while (count<n_features)
    {
        flag=0;
        temp=rand()%(col-1);
        for (int i = 0; i < count; i++)
            if (temp==featurelist[i])
                flag=1;
        if (flag==0)
        {
            featurelist[count]=temp;
            count++;
        } 
    }
    // 计算所有切分点Gini系数，选出Gini系数最小的切分点
    for (int i = 0; i < n_features; i++)
    {
        for (int j = 0; j < row; j++)
        {
            double value=dataset[j][featurelist[i]];
            score = gini_index(featurelist[i], value, row, col, dataset, class, classnum);
            if (score<b_score)
            {
                b_score = score;
                b_value = value;
                b_index = featurelist[i];
            }
        }
    }
    tree->index=b_index;tree->value=b_value;tree->flag=0;
    return tree;
}
```

该步骤其他代码与CART的代码一致，不再重复给出。

#### 2.2 随机森林算法

由2.1的步骤得到单棵决策树，将森林结果保存为结构体数组，并返回结构体二重指针。其中每棵树是用随机采样的训练集训练出来的。

```{c}
struct treeBranch **random_forest(int row, int col, double **data, int min_size, int max_depth, int n_features, int n_trees, float sample_size)
{
    struct treeBranch ** forest = (struct treeBranch **)malloc(n_trees * sizeof(struct treeBranch*));
    int samplenum = (int)(row*sample_size);
    int temp;
    // 生成随机训练集
    double ** subsample=(double **)malloc(samplenum * sizeof(double *));
    for (int i = 0; i < samplenum; i++)
    {
        subsample[i]=(double *)malloc(col * sizeof(double));
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
        forest[j]=tree;
    }
    return forest;
}
```

### 3.12.3 算法代码

我们现在知道了如何实现**随机森林算法**，那么我们把它应用到[声纳数据集 sonar.csv](https://aistudio.baidu.com/aistudio/datasetdetail/105756/0)

我们给出链接：https://aistudio.baidu.com/aistudio/datasetdetail/105756/0

#### C语言细节讲解

本节假设您已下载数据集 `sonar.csv`，并且它在当前工作目录中可用。下面我们给出一个完整实例，使用C语言详细讲解每一处细节。我们给出每一个.c文件的所有代码：

##### 1) read_csv.c

该步骤代码与前面CART部分相似，不再重复给出。

##### 2) k_fold.c

该步骤代码与前面CART部分相似，不再重复给出。

##### 3) RF.c

```c
#include "RF.h"

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
        flag = 0;
        temp = rand() % (col - 1);
        for (int i = 0; i < count; i++)
            if (temp == featurelist[i])
                flag = 1;
        if (flag == 0)
        {
            featurelist[count] = temp;
            count++;
        }
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
        for (int j = 0; i < count; i++)
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
```

##### 4) RF.h

```c
#ifndef RF
#define RF

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


```{c}
#include "RF.h"

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

```{c}
#include "RF.h"

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
        printf("score[%d] = %f%%\n", i, score[i]);
        free(split_copy);
    }
    double total = 0.0;
    for (l = 0; l < n_folds; l++)
    {
        total += score[l];
    }
    printf("mean_accuracy = %f%%\n", total / n_folds);
    return score;
}
```

##### 8) main.c

```C
#include "RF.h"

int main()
{
    char filename[] = "sonar.csv";
    char line[1024];
    row = get_row(filename);
    col = get_col(filename);
    dataset = (double **)malloc(row * sizeof(int *));
    for (int i = 0; i < row; ++i)
    {
        dataset[i] = (double *)malloc(col * sizeof(double));
    } //动态申请二维数组
    get_two_dimension(line, dataset, filename);

    // 输入模型参数，包括每个叶子最小样本数、最大层数、特征值选取个数、树木个数
    int min_size = 2, max_depth = 10, n_features = 7, n_trees = 100;
    double sample_size = 1;
    int n_folds = 5;
    int fold_size = (int)(row / n_folds);

    // 随机森林算法，返回交叉验证正确率
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
score[0] = 73.170732%
score[1] = 68.292683%
score[2] = 58.536585%
score[3] = 68.292683%
score[4] = 65.853659%
mean_accuracy = 66.829268%
```

#### Python语言实战

本节同样假设您已经下载数据集，我们使用著名机器学习开源库sklearn高效实现**随机森林算法**，以便您在实战中使用该算法：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
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
        model = RandomForestClassifier()
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
score[0] = 0.8333333333333334%
score[1] = 0.6666666666666666%
score[2] = 0.8095238095238095%
score[3] = 0.8048780487804879%
score[4] = 0.8780487804878049%
mean_accuracy = 0.7984901277584203%
```

