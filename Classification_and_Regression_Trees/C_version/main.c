#include "DT.h"

int main()
{
    char filename[] = "banknote.csv";
    char line[1024];
    row = get_row(filename);
    col = get_col(filename);
    dataset = (double **)malloc(row * sizeof(int *));
    for (int i = 0; i < row; ++i)
    {
        dataset[i] = (double *)malloc(col * sizeof(double));
    } //动态申请二维数组
    get_two_dimension(line, dataset, filename);

    // CART参数，分别为叶节点最小样本数和树最大层数
    int min_size = 5, max_depth = 10;
    int n_folds = 5;
    int fold_size = (int)(row / n_folds);

    // CART决策树，返回交叉验证正确率
    float* score = evaluate_algorithm(dataset, col, n_folds, fold_size, min_size, max_depth);
}