#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);

void main(){
	char filename[] = "seeds_data.csv";
    char line[1024];
    int row = get_row(filename);
    int col = get_col(filename);
    printf("row = %d\n",row);
    printf("col = %d\n",col);
    double **dataset = (double **)malloc(row*sizeof(int *));
    int i;
	for (i = 0; i < row; ++i){
		dataset[i] = (double *)malloc(col*sizeof(double));
	}//动态申请二维数组	
	get_two_dimension(line, dataset, filename);
    double l_rate = 0.1;
	int n_epoch = 100;
	int n_folds = 4;
	int fold_size;
    fold_size=(int)(row/n_folds);
	evaluate_algorithm(dataset, n_folds, fold_size, l_rate, n_epoch,col,row);
}
