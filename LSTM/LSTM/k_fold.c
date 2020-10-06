#include <stdlib.h>
#include <stdio.h>

double  ***cross_validation_split(double **dataset, int row, int n_folds, int fold_size,int col)
{
    srand(10);//种子
    double ***split;
    int i,j=0,k=0;
    int index;
    split=(double***)malloc(n_folds*sizeof(double**));
    for(i=0;i<n_folds;i++)
    {
        split[i] = (double**)malloc(fold_size * sizeof(double*));
        while(j<fold_size)
        {
            split[i][j] = (double*)malloc(col * sizeof(double));
            index=rand()%row;
            split[i][j] = dataset[index];
            
            for(k=index;k<row-1;k++)//for循环删除这个数组中被rand取到的元素
            {
                dataset[k]=dataset[k+1];
            }
            row--;//每次随机取出一个后总行数-1，保证不会重复取某一行
            j++;
        }
        j=0;//清零j
    }
    return split;
}
