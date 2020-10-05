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
