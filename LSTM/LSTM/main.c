#include "math.h"
#include "stdlib.h"
#include "time.h"
#include "assert.h"
#include "string.h"
#include "stdio.h" 
 

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern double* evaluate_algorithm(double** dataset, int n_folds, int fold_size, double l_rate, int n_epoch, int col, int row);

void main(){
	char filename[] = "data.csv";
    char line[1024];
    int row = get_row(filename);
    int col = get_col(filename);
    printf("row = %d\n",row);
    printf("col = %d\n",col);
    double **dataset = (double **)malloc(row*sizeof(double*));
    int i;
	for (i = 0; i < row; ++i){
		dataset[i] = (double *)malloc(col*sizeof(double));
	}
	//动态申请二维数组	
	get_two_dimension(line, dataset, filename);
    double l_rate = 0.01;
	int n_epoch = 100;
	int n_folds = 4;
	int fold_size;
    fold_size=(int)(row/n_folds);
	evaluate_algorithm(dataset, n_folds, fold_size, l_rate, n_epoch,col,row);
}


