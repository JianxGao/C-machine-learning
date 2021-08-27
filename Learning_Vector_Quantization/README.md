## 3.8 Learning Vector Quantization

### 3.8.1 算法简介

学习向量量化（Learning Vector Quantization）与K-Mean算法类似，其为试图找到一组原型向量来刻画聚类结构，但与一般的聚类算法不同的是，LVQ假设数据样本带有类别标记，学习过程利用样本的这些监督信息来辅助聚类，从而克服自组织网络采用无监督学习算法带来的缺乏分类信息的弱点。

向量量化的思路是，将高维输入空间分成若干个不同的区域，对每个区域确定一个中心向量作为聚类中心，与其处于同一个区域的输入向量可用作该中心向量来代表，从而形成了以各中心向量为聚类中心的点集。

#### LVQ网络结构与工作原理

其结构分为输入层、竞争层、输出层，竞争层和输出层之间完全连接。输出层每个神经元只与竞争层中的一组神经元连接，连接权重固定为1，训练过程中输入层和竞争层之间的权值逐渐被调整为聚类中心。当一个样本输入LVQ网络时，竞争层的神经元通过胜者为王学习规则产生获胜神经元，容许其输出为1，其它神经元输出为0。与获胜神经元所在组相连的输出神经元输出为1，而其它输出神经元为0，从而给出当前输入样本的模式类。将竞争层学习得到的类成为子类，而将输出层学习得到的类成为目标类。

#### LVQ网络学习算法

LVQ网络的学习规则结合了竞争学习规则和有导师学习规则，所以样本集应当为{(xi，di)}。其中di为l维，对应输出层的l个神经元，它只有一个分量为1，其他分量均为0。通常把竞争层的每个神经元指定给一个输出神经元，相应的权值为1，从而得到输出层的权值。比如某LVQ网络竞争层6个神经元，输出层3个神经元，代表3类。若将竞争层的1，3指定为第一个输出神经元，2，5指定为第二个输出神经元，3，6指定为第三个输出神经元。

训练前预先定义好竞争层到输出层权重，从而指定了输出神经元类别，训练中不再改变。网络的学习通过改变输入层到竞争层的权重来进行。根据输入样本类别和获胜神经元所属类别，可判断当前分类是否正确。若分类正确，则将获胜神经元的权向量向输入向量方向调整，分类错误则向相反方向调整。

### 3.8.2 算法讲解

#### 算法流程

输入：样本集$D=(x_1,y_1),(x_2,y_2)...(x_m,y_m)$;原型向量个数为q，各原型向量预设的类别标记$t_1,t_2...t_q$,学习率$\delta\in(0,1)$

1.初始化一些原型向量$p_1,p_2...p_q$

2.repeat

3.从样本集D随机选取样本$(x_j,y_j)$

4.计算样本$x_j$与$p_i(1<i<q)$

5.找出与$x_j$距离最近的原型向量$p_i$,$i^*=argmin_{i\in(1,2,...,q)}d_{ji}$

6.if $y_i=t_{i^*}$ ,then

7.$p'=p_{i^*}+\delta(x_j-p_{i^*})$

8、else

9.$p'=p_{i^*}-\delta(x_j-p_{i^*})$

10.end if

11.将原型向量$p_{i^*}$更新为$p'$

12.until满足停止条件

输出：原型向量$p_1, p_2,...,p_q$

#### 核心思想


1.对原型向量进行迭代优化，每一轮随机选择一个有标记的训练样本，找出与其距离最近的原型向量，根据两者的类别标记是否一致来对原型向量进行相应的更新。

2.LVQ的关键在于第6-10行如何更新原型向量，对于样本


$x_j$，若最近的原型向量$p_{i^*}$与$x_j$的类别标记相同，则令$p_{i^*}$向$x_j$方向靠近，否则远离其方向，学习率为$\delta$


#### 计算欧式距离

```c
double euclidean_distance(double*row1, double*row2){
    int i;
    double distance = 0.0;
    for (i=0;i<col-1;i++){
        distance =distance+ (row1[i] - row2[i])*(row1[i] - row2[i]);
    }
    return sqrt(distance);
    //其返回的是两个标志的欧氏距离的绝对值
}
```

```c
input:
row1:	2	4
row2:	1	3
output:	1
```

#### 2.4  确定最佳匹配位置

```c
int get_best_matching_unit(double**codebooks, double*test_row,int n_codebooks){
    double dist_min,dist;
    int i,min=0;
    dist_min = euclidean_distance(codebooks[0], test_row);
    for (i=0;i< n_codebooks;i++){
        dist = euclidean_distance(codebooks[i], test_row);
        if(dist < dist_min){
            dist_min=dist;
            min=i;
        }
    }
    //bmu=codebooks[min];
    return min;
}//其返回欧氏距离最小的标志的index
```

```c
input:
3	6	7	2
output:
3
```

#### 2.5  初始化原型向量

```c
double** random_codebook(double**train,int n_codebooks,int fold_size){
    int i,j,r;
    int n_folds=(int)(row/fold_size);
    double **codebooks=(double **)malloc(n_codebooks * sizeof(int*));
    for ( i=0;i < n_codebooks; ++i){
        codebooks[i] = (double *)malloc(col * sizeof(double));
    };
    srand((unsigned)time(NULL));
    for(i=0;i<n_codebooks;i++){
        for(j=0;j<col;j++){
            //srand((unsigned int)time(0));
            r=rand()%((n_folds-1)*fold_size);
            //printf(" r%d",r);
            codebooks[i][j]=train[r][j];
        }
    }
    return codebooks;
}//产生初始化原型向量
```

```c
output:
0.001251
0.563585
0.193304
0.808741
0.585009
0.479873
0.350291
0.895962
0.822840
0.746605
0.174108
0.858943
0.710501
0.513535
0.303995
0.014985
0.091403
0.364452
0.147313
0.165899
0.988525
0.445692
0.119083
0.004669
......
```

#### 预测神经网络

```c
float* get_test_prediction(double **train, double **test, float l_rate, int n_epoch, int fold_size)
{
    int i;
    double **codebooks=(double **)malloc(n_codebooks * sizeof(int*));
    for ( i=0;i < n_codebooks; ++i){
codebooks[i] = (double *)malloc(col * sizeof(double));
	};
    float *predictions=(float*)malloc(fold_size*sizeof(float));//预测集的行数就是数组prediction的长度

    codebooks=train_codebooks(train,l_rate,n_epoch,n_codebooks,fold_size);
    for(i=0;i<fold_size;i++)
    {
        predictions[i]=predict(codebooks,test[i]);
    }
	return predictions;
}//返回聚类预测分类结果
```

```c
output:
2
2
1
1
2
1
0
0
1
2
2
2
0
2
......
```

### 3.8.3 算法代码

我们现在知道了如何实现**学习向量量化算法**，那么我们把它应用到[电离数据集 ionosphere-full.csv](https://aistudio.baidu.com/aistudio/datasetdetail/105756/0)

我们给出链接：https://aistudio.baidu.com/aistudio/datasetdetail/105756/0

#### C语言细节讲解

本节假设您已下载数据集 `ionosphere-full.csv`，并且它在当前工作目录中可用。下面我们给出一个完整实例，使用C语言详细讲解每一处细节。我们给出每一个.c文件的所有代码：

##### 1) read_csv.c

该文件代码与前面代码一致，不再重复给出。

##### 2) k_fold.c

该文件代码与前面代码一致，不再重复给出。

##### 3) score.c

该文件代码与前面代码一致，不再重复给出。

##### 4) test_prediction.c

```c
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

extern double predict(int col, double **codebooks, double *test_row, int n_codebooks);
extern double **train_codebooks(double **train, int row, int col, double l_rate, int n_epoch, int n_codebooks, int fold_size);

double *get_test_prediction(double **train, double **test, int row, int col, double l_rate, int n_epoch, int fold_size, int n_codebooks)
{
    int i;
    double **codebooks = (double **)malloc(n_codebooks * sizeof(int *));
    for (i = 0; i < n_codebooks; ++i)
    {
        codebooks[i] = (double *)malloc(col * sizeof(double));
    };
    double *predictions = (double *)malloc(fold_size * sizeof(double)); //预测集的行数就是数组prediction的长度
    codebooks = train_codebooks(train, row, col, l_rate, n_epoch, n_codebooks, fold_size);
    for (i = 0; i < fold_size; i++)
    {
        predictions[i] = predict(col, codebooks, test[i], n_codebooks);
    }
    return predictions;
}
```

##### 5) evaluate.c

```c
#include <stdlib.h>
#include <stdio.h>

extern double accuracy_metric(double *actual, double *predicted, int fold_size);
extern double ***cross_validation_split(double **dataset, int row, int n_folds, int fold_size, int col);
extern double *get_test_prediction(double **train, double **test, int row, int col, double l_rate, int n_epoch, int fold_size, int n_codebooks);

void evaluate_algorithm(double **dataset, int row, int col, int n_folds, int fold_size, double l_rate, int n_epoch, int n_codebooks)
{
    double ***split;
    split = cross_validation_split(dataset, row, n_folds, fold_size, col);
    int i, j, k, l;
    int test_size = fold_size;
    int train_size = fold_size * (n_folds - 1);
    double *score = (double *)malloc(n_folds * sizeof(double));
    for (i = 0; i < n_folds; i++)
    {
        double ***split_copy = (double ***)malloc(n_folds * sizeof(int **));
        for (j = 0; j < n_folds; j++)
        {
            split_copy[j] = (double **)malloc(fold_size * sizeof(int *));
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
        double **test_set = (double **)malloc(test_size * sizeof(int *));
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
        double **train_set = (double **)malloc(train_size * sizeof(int *));
        for (k = 0; k < n_folds - 1; k++)
        {
            for (l = 0; l < fold_size; l++)
            {
                train_set[k * fold_size + l] = (double *)malloc(col * sizeof(double));
                train_set[k * fold_size + l] = split_copy[k][l];
            }
        }
        double *predicted = (double *)malloc(test_size * sizeof(double));
        predicted = get_test_prediction(train_set, test_set, row, col, l_rate, n_epoch, fold_size, n_codebooks);
        double *actual = (double *)malloc(test_size * sizeof(double));
        for (l = 0; l < test_size; l++)
        {
            actual[l] = (double)test_set[l][col - 1];
        }
        double accuracy = accuracy_metric(actual, predicted, test_size);
        score[i] = accuracy;
        printf("score[%d]=%.2f%%\n", i, score[i]);
        free(split_copy);
    }
    double total = 0.0;
    for (l = 0; l < n_folds; l++)
    {
        total += score[l];
    }
    printf("mean_accuracy=%.2f%%\n", total / n_folds);
}
```

##### 6) main.c

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <math.h>

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern void evaluate_algorithm(double **dataset, int row, int col, int n_folds, int fold_size, double l_rate, int n_epoch, int n_codebooks);

double euclidean_distance(int col, double *row1, double *row2)
{
    int i;
    double distance = 0.0;
    for (i = 0; i < col - 1; i++)
    {
        distance = distance + (row1[i] - row2[i]) * (row1[i] - row2[i]);
    }
    return sqrt(distance);
}

//Locate the best matching unit
int get_best_matching_unit(int col, double **codebooks, double *test_row, int n_codebooks)
{
    double dist_min, dist;
    int i, min = 0;
    dist_min = euclidean_distance(col, codebooks[0], test_row);
    for (i = 0; i < n_codebooks; i++)
    {
        dist = euclidean_distance(col, codebooks[i], test_row);
        if (dist < dist_min)
        {
            dist_min = dist;
            min = i;
        }
    }
    return min;
}

// Make a prediction with codebook vectors
double predict(int col, double **codebooks, double *test_row, int n_codebooks)
{
    int min;
    min = get_best_matching_unit(col, codebooks, test_row, n_codebooks);
    return (double)codebooks[min][col - 1];
}

// Create random codebook vectors
double **random_codebook(double **train, int row, int col, int n_codebooks, int fold_size)
{
    int i, j, r;
    int n_folds = (int)(row / fold_size);
    double **codebooks = (double **)malloc(n_codebooks * sizeof(int *));
    for (i = 0; i < n_codebooks; ++i)
    {
        codebooks[i] = (double *)malloc(col * sizeof(double));
    };
    srand((unsigned)time(NULL));
    for (i = 0; i < n_codebooks; i++)
    {
        for (j = 0; j < col; j++)
        {
            r = rand() % ((n_folds - 1) * fold_size);
            codebooks[i][j] = train[r][j];
        }
    }
    return codebooks;
}

double **train_codebooks(double **train, int row, int col, double l_rate, int n_epoch, int n_codebooks, int fold_size)
{
    int i, j, k, min = 0;
    double error, rate = 0.0;
    int n_folds = (int)(row / fold_size);
    double **codebooks = (double **)malloc(n_codebooks * sizeof(int *));
    for (i = 0; i < n_codebooks; ++i)
    {
        codebooks[i] = (double *)malloc(col * sizeof(double));
    };
    codebooks = random_codebook(train, row, col, n_codebooks, fold_size);
    for (i = 0; i < n_epoch; i++)
    {
        rate = l_rate * (1.0 - (i / (double)n_epoch));
        for (j = 0; j < fold_size * (n_folds - 1); j++)
        {
            min = get_best_matching_unit(col, codebooks, train[j], n_codebooks);
            for (k = 0; k < col - 1; k++)
            {
                error = train[j][k] - codebooks[min][k];
                if (fabs(codebooks[min][col - 1] - train[j][col - 1]) < 1e-13)
                {
                    codebooks[min][k] = codebooks[min][k] + rate * error;
                }
                else
                {
                    codebooks[min][k] = codebooks[min][k] - rate * error;
                }
            }
        }
    }
    return codebooks;
}

int main()
{
    char filename[] = "ionosphere-full.csv";
    char line[1024];
    int row = get_row(filename);
    int col = get_col(filename);
    int i;
    double **dataset = (double **)malloc(row * sizeof(int *));
    for (i = 0; i < row; ++i)
    {
        dataset[i] = (double *)malloc(col * sizeof(double));
    }
    get_two_dimension(line, dataset, filename);
    int n_folds = 5;
    double l_rate = 0.3;
    int n_epoch = 50;
    int fold_size = (int)(row / n_folds);
    int n_codebooks = 20;
    evaluate_algorithm(dataset, row, col, n_folds, fold_size, l_rate, n_epoch, n_codebooks);
    return 0;
}
```

##### 7) compile.sh

```bash
gcc main.c read_csv.c normalize.c k_fold.c evaluate.c score.c test_prediction.c -o run -lm && ./run
```

**编译&运行：**

```bash
bash compile.sh
```

最终输出结果如下：

```c
score[0] = 91.4286%
score[1] = 90.0000%
score[2] = 87.1429%
score[3] = 81.4286%
score[4] = 87.1429%
mean_accuracy = 87.4286%
```

#### Python语言实战

本节同样假设您已经下载数据集，我们使用著名机器学习开源库sklearn与sklearn_lvq高效实现**学习向量量化算法**，以便您在实战中使用该算法：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn_lvq import LgmlvqModel


if __name__ == '__main__':
    dataset = np.array(pd.read_csv("ionosphere-full.csv", sep=',', header=None))
    k_Cross = KFold(n_splits=5, random_state=8, shuffle=True)
    index = 0
    score = np.array([])
    data,label = dataset[:,:-1],dataset[:,-1]
    for train_index, test_index in k_Cross.split(dataset):
        train_data, train_label = data[train_index, :], label[train_index]
        test_data, test_label = data[test_index, :], label[test_index]
        model = LgmlvqModel()
        model.fit(train_data,train_label)
        pred = model.predict(test_data)
        acc = accuracy_score(test_label, pred)
        score = np.append(score,acc)
        print('score[{}] = {}%'.format(index,acc))
        index+=1
    print('mean_accuracy = {}%'.format(np.mean(score)))
```

输出结果如下：

```python
score[0] = 0.9014084507042254%
score[1] = 0.8714285714285714%
score[2] = 0.8428571428571429%
score[3] = 0.9%
score[4] = 0.9285714285714286%
mean_accuracy = 0.8888531187122737%
```

