## LVQ

### 1.算法简介

​	学习向量量化（Learning Vector Quantization）与K-Mean算法类似，其为试图找到一组原型向量来刻画聚类结构，但与一般的聚类算法不同的是，LVQ假设数据样本带有类别标记，学习过程利用样本的这些监督信息来辅助聚类，从而克服自组织网络采用无监督学习算法带来的缺乏分类信息的弱点。

​	向量量化的思路是，将高维输入空间分成若干个不同的区域，对每个区域确定一个中心向量作为聚类中心，与其处于同一个区域的输入向量可用作该中心向量来代表，从而形成了以各中心向量为聚类中心的点集。

#### 1.1 LVQ网络结构与工作原理

​	其结构分为输入层、竞争层、输出层，竞争层和输出层之间完全连接。输出层每个神经元只与竞争层中的一组神经元连接，连接权重固定为1，训练过程中输入层和竞争层之间的权值逐渐被调整为聚类中心。当一个样本输入LVQ网络时，竞争层的神经元通过胜者为王学习规则产生获胜神经元，容许其输出为1，其它神经元输出为0。与获胜神经元所在组相连的输出神经元输出为1，而其它输出神经元为0，从而给出当前输入样本的模式类。将竞争层学习得到的类成为子类，而将输出层学习得到的类成为目标类。

#### 1.2 LVQ网络学习算法

​	LVQ网络的学习规则结合了竞争学习规则和有导师学习规则，所以样本集应当为{(xi，di)}。其中di为l维，对应输出层的l个神经元，它只有一个分量为1，其他分量均为0。通常把竞争层的每个神经元指定给一个输出神经元，相应的权值为1，从而得到输出层的权值。比如某LVQ网络竞争层6个神经元，输出层3个神经元，代表3类。若将竞争层的1，3指定为第一个输出神经元，2，5指定为第二个输出神经元，3，6指定为第三个输出神经元。

​	训练前预先定义好竞争层到输出层权重，从而指定了输出神经元类别，训练中不再改变。网络的学习通过改变输入层到竞争层的权重来进行。根据输入样本类别和获胜神经元所属类别，可判断当前分类是否正确。若分类正确，则将获胜神经元的权向量向输入向量方向调整，分类错误则向相反方向调整。

#### 1.3 算法流程

输入：样本集$D=(x_1,y_1),(x_2,y_2)...(x_m,y_m)$;原型向量个数为q，各原型向量预设的类别标记$t_1,t_2...t_q$,学习率$\delta\in(0,1)$

1.初始化一些原型向量$p_1,p_2...p_q$

2.repeat

3.从样本集D随机选取样本$(x_j,y_j)$

4.计算样本$x_j$与$p_i(1<i<q)$的距离：$d_{ji}=||x_j-p_i||_2$

5.找出与$x_j$距离最近的原型向量$p_i$,$i^*=argmin_{i\in(1,2,...,q)}d_{ji}$

6.if $y_j=t_{i^*}$ ,then

7. $p'=p_{i^*}+\delta(x_j-p_{i^*})$

8.else

9. $p'=p_{i^*}-\delta(x_j-p_{i^*})$

10.end if

11.将原型向量$p_{i^*}$更新为$p'$

12.until满足停止条件

输出：原型向量$p_1,p_2,...,p_q$

#### 1.4 核心思想

​	1.对原型向量进行迭代优化，每一轮随机选择一个有标记的训练样本，找出与其距离最近的原型向量，根据两者的类别标记是否一致来对原型向量进行相应的更新。

​	2.LVQ的关键在于第6-10行如何更新原型向量，对于样本$x_j$，若最近的原型向量$p_{i^*}$与$x_j$的类别标记相同，则令$p_{i^*}$向$x_j$方向靠近，否则远离其方向，学习率为$\delta$.

#### 1.5 数据集

​	电离层数据集（Ionosphere Dataset）需要根据给定的电离层中的自由电子的雷达回波预测大气结构。它是一个二元分类问题。每个类的观察值数量不均等，一共有 351 个观察值，34 个输入变量和1个输出变量。

### 2.算法讲解

#### 	2.1  读取csv文件

​	该步骤代码与前面代码一致，不再重复给出。

#### 	2.2  数据划分

 该代码与公共代码一致，不再重复给出。

#### 2.3  计算欧式距离

```c
double euclidean_distance(double*row1, double*row2){
    int i;
    double distance = 0.0;
    for (i=0;i<col-1;i++){
        distance =distance+ (row1[i] - row2[i])*(row1[i] - row2[i]);
    }
    return sqrt(distance);
}//其返回的是两个标志的欧氏距离的绝对值
```

```c
input:
row1:2	4	
row2:1	3
output:
1.414214
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

#### 2.6  算法评估

```c
float* evaluate_algorithm(double **dataset, int n_folds, int fold_size, float l_rate, int n_epoch)//float 改为 float*
{
	double*** split ;//float*** 改为 double***
	split=  cross_validation_split(dataset,row, n_folds, fold_size);
	int i, j, k, l;
	int test_size = fold_size;
	int train_size = fold_size * (n_folds - 1);//train_size个一维数组
	float* score = (float*)malloc(n_folds * sizeof(float));
	for (i = 0; i < n_folds; i++)
    {  //因为要遍历删除，所以拷贝一份split
		double*** split_copy = (double***)malloc(n_folds * sizeof(int**));//float*** 改为 double***,float**改int**
		for (j = 0; j < n_folds; j++) {
			split_copy[j] = (double**)malloc(fold_size * sizeof(int*));//float** 改为 double**,float*改int*
			for (k = 0; k < fold_size; k++) {
				split_copy[j][k] = (double*)malloc(col * sizeof(double));//float* 改为 double*
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
		double** test_set = (double**)malloc(test_size * sizeof(int*));//float** 改为 double**,float*改int*
		for (j = 0; j < test_size; j++) {//对test_size中的每一行
			test_set[j] = (double*)malloc(col * sizeof(double));//float* 改为 double*
			for (k = 0; k < col; k++) {
				test_set[j][k] = split_copy[i][j][k];
			}
		}
		for (j = i; j < n_folds - 1; j++) {
			split_copy[j] = split_copy[j + 1];
		}
		double** train_set = (double**)malloc(train_size * sizeof(int*));//float** 改为 double**,float*改int*
		for (k = 0; k < n_folds - 1; k++) {
			for (l = 0; l < fold_size; l++) {
				train_set[k*fold_size + l] = (double*)malloc(col * sizeof(double));//float* 改为 double*
				train_set[k*fold_size + l] = split_copy[k][l];
			}
		}
		float* predicted = (float*)malloc(test_size * sizeof(float));//predicted有test_size个
		predicted = get_test_prediction(train_set, test_set, l_rate, n_epoch, fold_size);

		float* actual = (float*)malloc(test_size * sizeof(float));
		for (l = 0; l < test_size; l++) {
			actual[l] = (float)test_set[l][col - 1];
		}
		float accuracy = accuracy_metric(actual, predicted, test_size);
		score[i] = accuracy;
		printf("score[%d]=%.2f%%\n", i, score[i]);    //修改了输出格式
		free(split_copy);
	}
	float total = 0.0;
	for (l = 0; l < n_folds; l++) {
		total += score[l];
	}
	printf("mean_accuracy=%.2f%%\n", total / n_folds); //修改了输出格式
	return score;
}
```

```c
output:
score[0]=90.00%
score[1]=90.00%
score[2]=92.86%
score[3]=91.43%
score[4]=88.57%
mean_accuracy=90.57%
```

#### 		2.7  预测神经网络

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



### 		3.算法代码

#### 3.1  lib.c

```c
//对原公共模块进行了修改

#include <stdio.h>
#include <string.h>
#include <malloc.h>


double **dataset;
int row,col;

extern double* get_test_prediction(double **train, double **test, float l_rate, int n_epoch, int fold_size);


int get_row(char *filename)//获取行数 //未修改
{
	char line[1024];
	int i = 0;
	FILE* stream = fopen(filename, "r");
	while(fgets(line, 1024, stream)){
		i++;
	}
	fclose(stream);
	return i;
}

int get_col(char *filename)//获取列数   //未修改
{
	char line[1024];
	int i = 0;
	FILE* stream = fopen(filename, "r");
	fgets(line, 1024, stream);
	char* token = strtok(line, ",");
	while(token){
		token = strtok(NULL, ",");
		i++;
	}
	fclose(stream);
	return i;
}

void get_two_dimension(char *line, double **dataset, char *filename)    //未修改
{
	FILE* stream = fopen(filename, "r");
	int i = 0;
	while (fgets(line, 1024, stream))//逐行读取
    {
    	int j = 0;
    	char *tok;
        char *tmp = strdup(line);
        for (tok = strtok(line, ","); tok && *tok; j++, tok = strtok(NULL, ",\n")){
        	dataset[i][j] = atof(tok);//转换成浮点数
		}//字符串拆分操作
        i++;
        free(tmp);
    }
    fclose(stream);//文件打开后要进行关闭操作
}

float accuracy_metric(float *actual, float *predicted, int fold_size)
{
	int correct = 0;
	int i;
	for (i = 0; i < fold_size; i++)
    {
		if ((actual[i] - predicted[i])<10e-6)   //actual[i] == predicted[i] 改为 (actual[i] - predicted[i])<10e-6
			correct ++;
	}
	return (correct / (float)fold_size)*100.0;
}

double*** cross_validation_split(double **dataset, int row, int n_folds, int fold_size) //原声明是double，改为double***
{
    srand(10);//种子
    double*** split;
    int i,j=0,k=0;
    int index;
    double **fold;
    split=(double***)malloc(n_folds*sizeof(int**));//原本是double**，改为int**
    for(i=0;i<n_folds;i++)
    {
        fold = (double**)malloc(fold_size*sizeof(int *));//原本是double*，改为int*
        while(j<fold_size)
        {
            fold[j]=(double*)malloc(col*sizeof(double));
            index=rand()%row;
            fold[j]=dataset[index];
            for(k=index;k<row-1;k++)//for循环删除这个数组中被rand取到的元素
            {
                dataset[k]=dataset[k+1];
            }
            row--;//每次随机取出一个后总行数-1，保证不会重复取某一行
            j++;
        }
        j=0;//清零j
        split[i]=fold;
    }
    return split;
}


float* evaluate_algorithm(double **dataset, int n_folds, int fold_size, float l_rate, int n_epoch)//float 改为 float*
{
	double*** split ;//float*** 改为 double***
	split=  cross_validation_split(dataset,row, n_folds, fold_size);
	int i, j, k, l;
	int test_size = fold_size;
	int train_size = fold_size * (n_folds - 1);//train_size个一维数组
	float* score = (float*)malloc(n_folds * sizeof(float));
	for (i = 0; i < n_folds; i++)
    {  //因为要遍历删除，所以拷贝一份split
		double*** split_copy = (double***)malloc(n_folds * sizeof(int**));//float*** 改为 double***,float**改int**
		for (j = 0; j < n_folds; j++) {
			split_copy[j] = (double**)malloc(fold_size * sizeof(int*));//float** 改为 double**,float*改int*
			for (k = 0; k < fold_size; k++) {
				split_copy[j][k] = (double*)malloc(col * sizeof(double));//float* 改为 double*
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
		double** test_set = (double**)malloc(test_size * sizeof(int*));//float** 改为 double**,float*改int*
		for (j = 0; j < test_size; j++) {//对test_size中的每一行
			test_set[j] = (double*)malloc(col * sizeof(double));//float* 改为 double*
			for (k = 0; k < col; k++) {
				test_set[j][k] = split_copy[i][j][k];
			}
		}
		for (j = i; j < n_folds - 1; j++) {
			split_copy[j] = split_copy[j + 1];
		}
		double** train_set = (double**)malloc(train_size * sizeof(int*));//float** 改为 double**,float*改int*
		for (k = 0; k < n_folds - 1; k++) {
			for (l = 0; l < fold_size; l++) {
				train_set[k*fold_size + l] = (double*)malloc(col * sizeof(double));//float* 改为 double*
				train_set[k*fold_size + l] = split_copy[k][l];
			}
		}
		float* predicted = (float*)malloc(test_size * sizeof(float));//predicted有test_size个
		predicted = get_test_prediction(train_set, test_set, l_rate, n_epoch, fold_size);

		float* actual = (float*)malloc(test_size * sizeof(float));
		for (l = 0; l < test_size; l++) {
			actual[l] = (float)test_set[l][col - 1];
		}
		float accuracy = accuracy_metric(actual, predicted, test_size);
		score[i] = accuracy;
		printf("score[%d]=%.2f%%\n", i, score[i]);    //修改了输出格式
		free(split_copy);
	}
	float total = 0.0;
	for (l = 0; l < n_folds; l++) {
		total += score[l];
	}
	printf("mean_accuracy=%.2f%%\n", total / n_folds); //修改了输出格式
	return score;
}
```

#### 3.2  main.c

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <math.h>

int row,col,n_codebooks;

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern float evaluate_algorithm(double **dataset, int n_folds, int fold_size, float l_rate, int n_epoch);


double euclidean_distance(double*row1, double*row2){
    int i;
    double distance = 0.0;
    for (i=0;i<col-1;i++){
        distance =distance+ (row1[i] - row2[i])*(row1[i] - row2[i]);
    }
    return sqrt(distance);
}

//Locate the best matching unit
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
}

// Make a prediction with codebook vectors
float predict(double**codebooks, double*test_row){
    int min;
    min=get_best_matching_unit(codebooks,test_row,n_codebooks);
    return (float)codebooks[min][col-1];
}

// Create random codebook vectors
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
}

double** train_codebooks(double**train, double l_rate, int n_epoch,int n_codebooks,int fold_size){
    int i,j,k,min=0;
    double error,rate=0.0;
    int n_folds=(int)(row/fold_size);
    double **codebooks=(double **)malloc(n_codebooks * sizeof(int*));
    for ( i=0;i < n_codebooks; ++i){
		codebooks[i] = (double *)malloc(col * sizeof(double));
	};
    codebooks=random_codebook(train,n_codebooks,fold_size);

    for (i=0;i<n_epoch;i++){
        rate = l_rate * (1.0-(i/(double)n_epoch));
        //printf("%d ",i);
        for(j=0;j<fold_size*(n_folds-1);j++){
            //printf("%d ",i);
            min=get_best_matching_unit(codebooks, train[j],n_codebooks);

            for (k=0;k<col-1;k++){
                error = train[j][k] - codebooks[min][k];
                if (fabs(codebooks[min][col-1] - train[j][col-1])<1e-13){
                    codebooks[min][k] =codebooks[min][k]+ rate * error;}
                else{codebooks[min][k] = codebooks[min][k]-rate * error;}

                }
            }
        }

    return codebooks;
}

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
}


int main()
{
    char filename[] = "ionosphere-full.csv";
    char line[1024];
    row = get_row(filename);
    col = get_col(filename);

    int i;
	double **dataset = (double **)malloc(row * sizeof(int *));
    for ( i=0;i < row; ++i){
		dataset[i] = (double *)malloc(col * sizeof(double));
	}

    get_two_dimension(line,dataset,filename);

    int n_folds=5;
    double l_rate=0.3;
    int n_epoch=50;
    int fold_size=(int)(row/n_folds);
    n_codebooks = 20;

    evaluate_algorithm(dataset, n_folds, fold_size, l_rate, n_epoch);

    return 0;

}
```

