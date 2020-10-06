#include<stdlib.h>
#include<stdio.h>

extern float* get_test_prediction(int col, int row, float** train, float** test, int n_folds);
extern float rmse_metric(float *actual, float *predicted, int fold_size);
extern double***  cross_validation_split(double **dataset, int row, int col, int n_folds, int fold_size);

float* evaluate_algorithm(double **dataset, int row, int col,int n_folds) {
	int fold_size = (int)row / n_folds;
	double ***split = cross_validation_split(dataset, row, n_folds, fold_size, col);
	int i, j, k, l;	
	int test_size = fold_size;
	int train_size = fold_size * (n_folds - 1);
	float* score = (float*)malloc(n_folds * sizeof(float));
	for (i = 0; i < n_folds; i++) {
		//因为要遍历删除，所以拷贝一份split
		float*** split_copy = (float***)malloc(n_folds * sizeof(float**));
		for (j = 0; j < n_folds; j++) {
			split_copy[j] = (float**)malloc(fold_size * sizeof(float*));
			for (k = 0; k < fold_size; k++) {
				split_copy[j][k] = (float*)malloc(col* sizeof(float));
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

		float** test_set = (float**)malloc(test_size * sizeof(float*));
		for (j = 0; j < test_size; j++) {
			test_set[j] = (float*)malloc(col * sizeof(float));
			for (k = 0; k < col; k++) {
				test_set[j][k] = split_copy[i][j][k];
			}
		}
		for (j = i; j < n_folds - 1; j++) {
			split_copy[j] = split_copy[j + 1];
		}//删除取出来的fold
		
		float** train_set = (float**)malloc(train_size * sizeof(float*));
		for (k = 0; k < n_folds - 1; k++) {
			for (l = 0; l < fold_size; l++) {
				train_set[k*fold_size + l] = (float*)malloc(col * sizeof(float));
				train_set[k*fold_size + l] = split_copy[k][l];
				//printf("split_copy[%d][%d]=%f\n", k,l,split_copy[k][l]);
			}
		}

		float* predicted = (float*)malloc(test_size * sizeof(float));
		predicted = get_test_prediction(col, row,  train_set,  test_set,  n_folds);
		float* actual = (float*)malloc(test_size * sizeof(float));
		for (l = 0; l < test_size; l++) {
			actual[l] = test_set[l][col - 1];
		}
		float rmse = rmse_metric(actual, predicted, test_size);
		score[i] = rmse;
		printf("score[%d]=%f\n", i, score[i]);
		free(split_copy);
	}
	float total = 0;
	for (l = 0; l < n_folds; l++) {
		total += score[l];
	}
	printf("mean_rmse=%f\n", total / n_folds);
	return score;
}