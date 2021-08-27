#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern void normalize_dataset(double **dataset, int row, int col);
extern void evaluate_algorithm(double **dataset, int row, int col, int n_folds, int num_neighbors, double l_rate, int n_epoch);

void main()
{
	char filename[] = "sonar.csv";
	char line[1024];
	int row = get_row(filename);
	int col = get_col(filename);
	double **dataset;
	dataset = (double **)malloc(row * sizeof(double *));
	for (int i = 0; i < row; ++i)
	{
		dataset[i] = (double *)malloc(col * sizeof(double));
	}
	get_two_dimension(line, dataset, filename);
	normalize_dataset(dataset, row, col);
	int k_fold = 3;
	int num_neighbours = 2;
	double l_rate = 0.01;
	int n_epoch = 5000;
	evaluate_algorithm(dataset, row, col, k_fold, num_neighbours, l_rate, n_epoch);
}

// 手动编译
// gcc main.c normalize.c score.c test_prediction.c stacking_model.c k_fold.c knn_model.c evaluate.c read_csv.c perceptron_model.c -o test -lm