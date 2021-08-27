#include <stdio.h>
#include <stdlib.h>

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern void normalize_dataset(double **dataset, int row, int col);
extern void evaluate_algorithm(double **dataset, int row, int col, int n_folds, int m_passes, double tol, double c, double change_limit);

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
	int n_folds = 5;
	int max_passes = 1;
	double C = 0.05;
	double tolerance = 0.001;
	double change_limit = 0.001;
	evaluate_algorithm(dataset, row, col, n_folds, max_passes, tolerance, C, change_limit);
}