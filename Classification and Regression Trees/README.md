## Classification and Regression Trees

> ​		决策树是一种广受欢迎的、强大的预测方法。它之所以受到欢迎，是因为其最终的模型对于从业人员来说易于理解，给出的决策树可以确切解释为何做出特定的预测。决策树是最简单的机器学习算法，它易于实现，可解释性强，完全符合人类的直观思维，有着广泛的应用。
>
> ​		同时，决策树也是更为高级的集成算法（如bagging，random forests和gradient boosting等）的基础。在本节中，您将了解Gini指数的概念、如何创建数据集的拆分、如何构建一棵树、如何利用构建的树作出分类决策以及如何在Banknote数据集上应用这些知识。

### 1.算法介绍

​		Classification and Regression Trees（简称CART），指的是可用于分类或回归预测建模问题的决策树算法。在本节中，我们将重点介绍如何使用CART解决分类问题，并以Banknote数据集为例进行演示。

​		CART模型的表示形式是一棵二叉树。每个节点表示单个输入变量（X）和该变量的分割点（假定变量是数字化的）。树的叶节点（也称作终端节点）包含用于预测的输出变量（y）。		

​		创建二元决策树实际上是划分输入空间的过程。一般采用贪婪方法对变量进行递归的二进制拆分，使用某个成本函数（通常是Gini指数）测试不同的分割点，选择成本最高的拆分（即拆分完之后，剩余成本降到最低，亦代表这种拆分所含的“信息量”最大）。

### 2.算法讲解

#### 2.1 按属性分割数据

* 功能：切分函数，根据切分点将数据分为左右两组
* 输出：从切分点处切分后的数据结果

- ```c
    struct dataset *test_split(int index, double value, int row, int col, double **data)
    {
        // 将切分结果作为结构体返回
        struct dataset *split = (struct dataset *)malloc(sizeof(struct dataset));
        int count1=0,count2=0;
        double ***groups = (double ***)malloc(2 * sizeof(double **));
        for (int i = 0; i < 2; i++)
        {
            groups[i]=(double **)malloc(row * sizeof(double *));
            for (int j = 0; j < row; j++)
            {
                groups[i][j] = (double *)malloc(col * sizeof(double ));
            }
        }
        for (int i = 0; i < row; i++)
        {
            if (data[i][index]<value)
            {
                groups[0][count1]=data[i];
                count1 ++;
            }else{
                groups[1][count2] = data[i];
                count2++;
            }
        }
        split->splitdata = groups;
        split->row1 = count1;
        split->row2 = count2;
        return split;
    }
    ```

#### 2.2 Gini指数

​		基尼指数是用于评估数据集中的拆分所常用的成本函数。数据集中的拆分涉及一个输入属性和该属性的一个值。它可以用于将训练模式分为两组。最理想的拆分是使基尼指数变为0，而最坏的情况是在二分类问题中分为每一类的概率都是50%（即基尼指数变为0.5）。

​		基尼系数的具体计算公式如下：
$$
G = 1-\sum^{k}_{i=1}{p_i^2}\tag{5.1}
$$
​		其中$k$是数据集中样本分类的数量，$p_i$表示第$i$类样本占总样本的比例。如果某一属性取多个值，则按照每一个值所占的比重进行加权平均。

​		例如，对于下面这些样本：

| day  | deadline? | party? | lazy? | activity |
| ---- | --------- | ------ | ----- | -------- |
| 1    | urgent    | yes    | yes   | party    |
| 2    | urgent    | no     | yes   | study    |
| 3    | near      | yes    | yes   | party    |
| 4    | none      | yes    | no    | party    |
| 5    | none      | no     | yes   | pub      |
| 6    | none      | yes    | no    | party    |
| 7    | near      | no     | no    | study    |
| 8    | near      | no     | yes   | TV       |
| 9    | near      | yes    | yes   | party    |
| 10   | urgent    | no     | no    | study    |

​		以“deadline?”这个属性为例。首先计算deadline这个属性取每一个值的比例：
$$
P(deadline=urgent)={3\over10}\\
P(deadline=near)={4\over10}\\
P(deadline=none)={3\over10}\tag{5.2}
$$
​		然后分别计算deadline这个属性取每一个值下的Gini指数：
$$
P(deadline=urgent\&activity=party)={1\over3}\\
P(deadline=urgent\&activity=study)={2\over3}\\
G(urgent)=1-(({1\over3})^2+({2\over3})^2)={4\over9}\tag{5.3}
$$

$$
P(deadline=near\&activity=party)={2\over4}\\
P(deadline=near\&activity=study)={1\over4}\\
P(deadline=near\&activity=TV)={1\over4}\\
G(near)=1-(({2\over4})^2+({1\over4})^2+({1\over4})^2)={5\over8}\tag{5.4}
$$

$$
P(deadline=none\&activity=party)={2\over3}\\
P(deadline=none\&activity=pub)={1\over3}\\
G(none)=1-(({2\over3})^2+({1\over3})^2)={4\over9}\tag{5.5}
$$

​		最后按照取每一个值所占的比重对以上三个Gini指数做加权平均：
$$
G_1=G(deadline)={3\over10}\times{4\over9}+{4\over10}\times{5\over8}+{3\over10}\times{4\over9}={31\over60}\tag{5.6}
$$
​		同理可以算出按属性“party?”和“lazy?”切分时的Gini指数：
$$
G_2=G(party)={5\over10}\times[1-({5\over5})^2]+{5\over10}\times[1-(({3\over5})^2+({1\over5})^2+({1\over5})^2)]={7\over25}\tag{5.7}
$$

$$
G_3=G(lazy)={6\over10}\times[1-(({3\over6})^2+({1\over6})^2+({1\over6})^2+({1\over6})^2)]+{4\over10}\times[1-(({2\over4})^2+({2\over4})^2)]={3\over5}\tag{5.8}
$$

​		由于$G_2<G_1<G_3$，所以我们在第一层选择“party”这个属性进行拆分。在下一层重复该步骤即可。需要注意的是，在分下一层时，样本就减少了（因为有一些已经分出去了）。

```{c}
double gini_index(int index,double value,int row, int col, double **dataset, double *class, int classnum)
{
    float *numcount1 = (float *)malloc(classnum * sizeof(float));
    float *numcount2 = (float *)malloc(classnum * sizeof(float));
    for (int i = 0; i < classnum; i++)
        numcount1[i]=numcount2[i]=0;

    float count1 = 0, count2 = 0;
    double gini1,gini2,gini;
    gini1=gini2=gini=0;
    // 计算每一类的个数
    for (int i = 0; i < row; i++)
    {
        if (dataset[i][index] < value)
        {
            count1 ++;
            for (int j = 0; j < classnum; j++)
                if (dataset[i][col-1]==class[j])
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
    if (count1==0)
    {
        gini1=1;
        for (int i = 0; i < classnum; i++)
            gini2 += (numcount2[i] / count2) * (numcount2[i] / count2);
    }else if (count2==0)
    {
        gini2=1;
        for (int i = 0; i < classnum; i++)
            gini1 += (numcount1[i] / count1) * (numcount1[i] / count1);
    }else
    {
        for (int i = 0; i < classnum; i++)
        {
            gini1 += (numcount1[i] / count1) * (numcount1[i] / count1);
            gini2 += (numcount2[i] / count2) * (numcount2[i] / count2);
        }
    }
    // 计算Gini指数
    gini1 = 1 - gini1;
    gini2 = 1 - gini2;
    gini = (count1 / row) * gini1 + (count2 / row) * gini2;
    free(numcount1);free(numcount2);
    numcount1=numcount2=NULL;
    return gini;
}
```

#### 2.3 寻找最佳分割点

​		我们需要根据计算出的Gini指数来决定最佳的分割点。具体做法是计算所有切分点Gini指数，选出Gini指数最小的切分点作为最后的分割点。

* 功能：选取数据的最优切分点
* 输出：数据中最优切分点下的树结构

- ```c
    struct treeBranch *get_split(int row, int col, double **dataset, double *class, int classnum)
    {
        struct treeBranch *tree=(struct treeBranch *)malloc(sizeof(struct treeBranch));
        int b_index=999;
        double b_score = 999, b_value = 999,score;
        // 计算所有切分点Gini系数，选出Gini系数最小的切分点
        for (int i = 0; i < col-1; i++)
        {
            for (int j = 0; j < row; j++)
            {
                double value=dataset[j][i];
                score=gini_index(i,value,row,col,dataset,class,classnum);
                if (score<b_score)
                {
                    b_score=score;
                    b_value=value;
                    b_index=i;
                }
            }
        }
        tree->index=b_index;tree->value=b_value;tree->flag=0;
        return tree;
    }
    ```

#### 2.4 计算叶子节点结果

​		我们不能让树一直生长下去，为此我们一般有两种方法来决定何时停止树的生长。

1. 最大树深。这是从树的根节点开始的最大节点数。一旦达到树的最大深度，就必须停止添加新节点。更深的树更复杂，可能更适合训练数据。
2. 最小节点记录。这是给定节点负责的最少训练模式。一旦达到或低于此最小值，我们必须停止拆分和添加新节点。 训练模式很少的节点可能过于具体，可能会过度拟合训练数据。

​		当我们在某个点停止树的生长时，该节点称为终端节点或叶子节点，用于做出最终预测。这是通过获取分配给该节点的行组并选择该组中最常见的类值来完成的。下面这个函数将为一组行选择一个类值，它返回行列表中最常见的输出值。

* 功能：计算叶子节点结果

* 输出：输出最多的一类

* ```c
    double to_terminal(int row, int col, double **data, double *class, int classnum)
    {
        int *num=(int *)malloc(classnum*sizeof(classnum));
        double maxnum=0;
        int flag=0;
        // 计算所有样本中结果最多的一类
        for (int i = 0; i < classnum; i++)
            num[i]=0;
        for (int i = 0; i < row; i++)
            for (int j = 0; j < classnum; j++)
                if (data[i][col-1]==class[j])
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
        num=NULL;
        return maxnum;
    }
    ```

#### 2.5 分裂左右迭代

​		通过上述尝试，我们已经知道如何以及何时创建叶子节点。现在我们可以建立我们的树了。构建决策树需要在为每个节点创建的组上反复调用上面的get_split()函数。添加到现有节点的新节点称为子节点。一个节点可以有零个子节点（叶子节点），一个子节点（在某一侧直接进行预测）或两个子节点。

​		创建节点后，我们可以通过再次调用相同的函数在拆分后的每一组数据上递归创建子节点。下面是实现此递归过程的函数。

* 功能：创建子树或生成叶子节点

* 输出：生成子树或叶子节点后的树

* ```c
    void split(struct treeBranch *tree, int row, int col, double **data, double *class, int classnum, int depth, int min_size, int max_depth)
    {
        // 判断是否已经达到最大层数
        if (depth>=max_depth)
        {
            tree->flag=1;
            tree->output = to_terminal(row, col, data, class, classnum);
            return;
        }
        struct dataset *childdata = test_split(tree->index, tree->value, row, col, data);
        // 判断样本是否已被分为一边
        if (childdata->row1==0 || childdata->row2==0)
        {
            tree->flag = 1;
            tree->output = to_terminal(row, col, data, class, classnum);
            return;
        }
        // 左子树，判断样本是否达到最小样本数，如不是则继续迭代
        if (childdata->row1<=min_size)
        {
            struct treeBranch *leftchild = (struct treeBranch *)malloc(sizeof(struct treeBranch));
            leftchild->flag=1;
            leftchild->output = to_terminal(childdata->row1, col, childdata->splitdata[0], class, classnum);
            tree->leftBranch=leftchild;
        }
        else
        {
            struct treeBranch *leftchild = get_split(childdata->row1, col, childdata->splitdata[0], class, classnum);
            tree->leftBranch=leftchild;
            split(leftchild, childdata->row1, col, childdata->splitdata[0], class, classnum, depth+1, min_size, max_depth);
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
            struct treeBranch *rightchild = get_split(childdata->row2, col, childdata->splitdata[1], class, classnum);
            tree->rightBranch = rightchild;
            split(rightchild, childdata->row2, col, childdata->splitdata[1], class, classnum, depth + 1, min_size, max_depth);
        }
        free(childdata->splitdata);childdata->splitdata=NULL;
        free(childdata);childdata=NULL;
        return;
    }
    ```

#### 2.6 建立决策树

​		下面，我们就可以利用上面编写的函数构建根节点并调用split()函数，然后进行递归调用以构建整个树。

* 功能：生成决策树

* 输出：生成后的决策树

* ```c
    struct treeBranch *build_tree(int row, int col, double **data, int min_size, int max_depth)
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
        struct treeBranch *result = get_split(row, col, data, classes, count1);
        // 进入迭代，不断生成子树
        split(result, row, col, data, classes, count1, 1, min_size, max_depth);
        return result;
    }
    ```

### 3.算法实现步骤

下面将通过多C文件，以[banknote数据集](https://goo.gl/rLPrHO)为例，应用决策树算法。

#### 3.1 read_csv.c

该步骤代码与前面代码一致，不再重复给出。

#### 3.2 k_fold.c

该步骤代码与前面代码一致，不再重复给出。

#### 3.3 DT.c

核心函数部分，用以构建整棵决策树，并给出决策树的预测结果。

```{c}
#include "DT.h"

// 切分函数，根据切分点将数据分为左右两组
struct dataset *test_split(int index, double value, int row, int col, double **data)
{
    // 将切分结果作为结构体返回
    struct dataset *split = (struct dataset *)malloc(sizeof(struct dataset));
    int count1=0,count2=0;
    double ***groups = (double ***)malloc(2 * sizeof(double **));
    for (int i = 0; i < 2; i++)
    {
        groups[i]=(double **)malloc(row * sizeof(double *));
        for (int j = 0; j < row; j++)
        {
            groups[i][j] = (double *)malloc(col * sizeof(double ));
        }
    }
    for (int i = 0; i < row; i++)
    {
        if (data[i][index]<value)
        {
            groups[0][count1]=data[i];
            count1 ++;
        }else{
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
double gini_index(int index,double value,int row, int col, double **dataset, double *class, int classnum)
{
    float *numcount1 = (float *)malloc(classnum * sizeof(float));
    float *numcount2 = (float *)malloc(classnum * sizeof(float));
    for (int i = 0; i < classnum; i++)
        numcount1[i]=numcount2[i]=0;

    float count1 = 0, count2 = 0;
    double gini1,gini2,gini;
    gini1=gini2=gini=0;
    // 计算每一类的个数
    for (int i = 0; i < row; i++)
    {
        if (dataset[i][index] < value)
        {
            count1 ++;
            for (int j = 0; j < classnum; j++)
                if (dataset[i][col-1]==class[j])
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
    if (count1==0)
    {
        gini1=1;
        for (int i = 0; i < classnum; i++)
            gini2 += (numcount2[i] / count2) * (numcount2[i] / count2);
    }else if (count2==0)
    {
        gini2=1;
        for (int i = 0; i < classnum; i++)
            gini1 += (numcount1[i] / count1) * (numcount1[i] / count1);
    }else
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
    free(numcount1);free(numcount2);
    numcount1=numcount2=NULL;
    return gini;
}

// 选取数据的最优切分点
struct treeBranch *get_split(int row, int col, double **dataset, double *class, int classnum)
{
    struct treeBranch *tree=(struct treeBranch *)malloc(sizeof(struct treeBranch));
    int b_index=999;
    double b_score = 999, b_value = 999,score;
    // 计算所有切分点Gini系数，选出Gini系数最小的切分点
    for (int i = 0; i < col-1; i++)
    {
        for (int j = 0; j < row; j++)
        {
            double value=dataset[j][i];
            score=gini_index(i,value,row,col,dataset,class,classnum);
            if (score<b_score)
            {
                b_score=score;
                b_value=value;
                b_index=i;
            }
        }
    }
    tree->index=b_index;tree->value=b_value;tree->flag=0;
    return tree;
}

// 计算叶节点结果
double to_terminal(int row, int col, double **data, double *class, int classnum)
{
    int *num=(int *)malloc(classnum*sizeof(classnum));
    double maxnum=0;
    int flag=0;
    // 计算所有样本中结果最多的一类
    for (int i = 0; i < classnum; i++)
        num[i]=0;
    for (int i = 0; i < row; i++)
        for (int j = 0; j < classnum; j++)
            if (data[i][col-1]==class[j])
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
    num=NULL;
    return maxnum;
}

// 创建子树或生成叶节点
void split(struct treeBranch *tree, int row, int col, double **data, double *class, int classnum, int depth, int min_size, int max_depth)
{
    // 判断是否已经达到最大层数
    if (depth>=max_depth)
    {
        tree->flag=1;
        tree->output = to_terminal(row, col, data, class, classnum);
        return;
    }
    struct dataset *childdata = test_split(tree->index, tree->value, row, col, data);
    // 判断样本是否已被分为一边
    if (childdata->row1==0 || childdata->row2==0)
    {
        tree->flag = 1;
        tree->output = to_terminal(row, col, data, class, classnum);
        return;
    }
    // 左子树，判断样本是否达到最小样本数，如不是则继续迭代
    if (childdata->row1<=min_size)
    {
        struct treeBranch *leftchild = (struct treeBranch *)malloc(sizeof(struct treeBranch));
        leftchild->flag=1;
        leftchild->output = to_terminal(childdata->row1, col, childdata->splitdata[0], class, classnum);
        tree->leftBranch=leftchild;
    }
    else
    {
        struct treeBranch *leftchild = get_split(childdata->row1, col, childdata->splitdata[0], class, classnum);
        tree->leftBranch=leftchild;
        split(leftchild, childdata->row1, col, childdata->splitdata[0], class, classnum, depth+1, min_size, max_depth);
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
        struct treeBranch *rightchild = get_split(childdata->row2, col, childdata->splitdata[1], class, classnum);
        tree->rightBranch = rightchild;
        split(rightchild, childdata->row2, col, childdata->splitdata[1], class, classnum, depth + 1, min_size, max_depth);
    }
    free(childdata->splitdata);childdata->splitdata=NULL;
    free(childdata);childdata=NULL;
    return;
}

// 生成决策树
struct treeBranch *build_tree(int row, int col, double **data, int min_size, int max_depth)
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
    struct treeBranch *result = get_split(row, col, data, classes, count1);
    // 进入迭代，不断生成子树
    split(result, row, col, data, classes, count1, 1, min_size, max_depth);
    return result;
}

// 决策树预测
double predict(double *test,struct treeBranch *tree)
{
    double output;
    // 判断是否达到叶节点，flag=1时为叶节点，flag=0时则继续判断
    if (tree->flag==1)
    {
        output = tree->output;
        return output;
    }
    else
    {
        if (test[tree->index]<tree->value)
        {
            output = predict(test, tree->leftBranch);
            return output;
        }
        else
        {
            output = predict(test, tree->rightBranch);
            return output;
        } 
    }
}
```

#### 3.4 DT.h

构建决策树所需要包含的头文件。

```{c}
#ifndef DT
#define DT

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

double **dataset;
int row, col;

struct treeBranch
{
    int flag;
    int index;
    double value;
    double output;
    struct treeBranch *leftBranch;
    struct treeBranch *rightBranch;
};

struct dataset
{
    int row1;
    int row2;
    double ***splitdata;
};

struct dataset *test_split(int index, double value, int row, int col, double **data);
double gini_index(int index, double value, int row, int col, double **dataset, double *class, int classnum);
struct treeBranch *get_split(int row, int col, double **dataset, double *class, int classnum);
double to_terminal(int row, int col, double **data, double *class, int classnum);
void split(struct treeBranch *tree, int row, int col, double **data, double *class, int classnum, int depth, int min_size, int max_depth);
struct treeBranch *build_tree(int row, int col, double **data, int min_size, int max_depth);
double predict(double *test, struct treeBranch *tree);

#endif
```

#### 3.5 test_prediction.c

​		我们将上述的预测在测试集上也进行一遍，由此判断模型对于没有见过的数据会做出怎样的预测，方便进一步对模型的好坏作出评估。

```{c}
double* get_test_prediction(double **train, double **test, int column, int min_size, int max_depth, int fold_size, int train_size)
{
	double *predictions = (double *)malloc(fold_size * sizeof(double)); //预测集的行数就是数组prediction的长度
	struct treeBranch *tree = build_tree(train_size, column, train, min_size, max_depth);
	for (int i = 0; i < fold_size; i++)
	{
		 predictions[i] = predict(test[i], tree);
	}
	return predictions; //返回对test的预测数组
}
```

#### 3.6 score.c

该步骤代码与前面代码一致，不再重复给出。

#### 3.7 evaluate.c

```{c}
float* evaluate_algorithm(double **dataset, int column, int n_folds, int fold_size, int min_size, int max_depth)
{
	double ***split = cross_validation_split(dataset, row, n_folds, fold_size);
	int i, j, k, l;
	int test_size = fold_size;
	int train_size = fold_size * (n_folds - 1); //train_size个一维数组
	float *score = (float *)malloc(n_folds * sizeof(float));
	for (i = 0; i < n_folds; i++)
	{ //因为要遍历删除，所以拷贝一份split
		double ***split_copy = (double ***)malloc(n_folds * sizeof(double**));
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
		predicted = get_test_prediction(train_set, test_set, column, min_size, max_depth, fold_size, train_size);
		double *actual = (double *)malloc(test_size * sizeof(double));
		for (l = 0; l < test_size; l++)
		{
			actual[l] = test_set[l][column - 1];
		}
		float accuracy = accuracy_metric(actual, predicted, test_size);
		score[i] = accuracy;
		printf("score[%d]=%f\n", i, score[i]);
		free(split_copy);
	}
	float total = 0.0;
	for (l = 0; l < n_folds; l++)
	{
		total += score[l];
	}
	printf("mean_accuracy=%f\n", total / n_folds);
	return score;
}
```

#### 3.8 main.c

```{c}
#include "DT.h"

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern double*** cross_validation_split(double **dataset, int row, int n_folds, int fold_size);
extern double* get_test_prediction(double **train, double **test, int column, int min_size, int max_depth, int fold_size, int train_size);
extern float accuracy_metric(double *actual, double *predicted, int fold_size);
extern float* evaluate_algorithm(double **dataset,int column, int n_folds, int fold_size, int min_size, int max_depth);

int main()
{
	char filename[] = "banknote.csv";
	char line[1024];
	row = get_row(filename);
	col = get_col(filename);
	dataset = (double **)malloc(row * sizeof(int *));
	for (int i = 0; i < row; ++i)
	{
		dataset[i] = (double *)malloc(col * sizeof(double));
	} //动态申请二维数组
	get_two_dimension(line, dataset, filename);

	// CART参数，分别为叶节点最小样本数和树最大层数
	int min_size = 5, max_depth = 10;
	int n_folds = 5;
	int fold_size = (int)(row / n_folds);

	// CART决策树，返回交叉验证正确率
	float* score = evaluate_algorithm(dataset, col, n_folds, fold_size, min_size, max_depth);
}
```