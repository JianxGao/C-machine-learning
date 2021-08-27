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