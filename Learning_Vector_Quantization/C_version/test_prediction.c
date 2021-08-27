#include <stdlib.h>
#include <stdio.h>
#include <math.h>

extern double predict(int col, double **codebooks, double *test_row, int n_codebooks);
extern double **train_codebooks(double **train, int row, int col, double l_rate, int n_epoch, int n_codebooks, int fold_size);

double *get_test_prediction(double **train, double **test, int row, int col, double l_rate, int n_epoch, int fold_size, int n_codebooks)
{
    int i;
    double **codebooks = (double **)malloc(n_codebooks * sizeof(int *));
    for (i = 0; i < n_codebooks; ++i)
    {
        codebooks[i] = (double *)malloc(col * sizeof(double));
    };
    double *predictions = (double *)malloc(fold_size * sizeof(double)); //预测集的行数就是数组prediction的长度
    codebooks = train_codebooks(train, row, col, l_rate, n_epoch, n_codebooks, fold_size);
    for (i = 0; i < fold_size; i++)
    {
        predictions[i] = predict(col, codebooks, test[i], n_codebooks);
    }
    return predictions;
}