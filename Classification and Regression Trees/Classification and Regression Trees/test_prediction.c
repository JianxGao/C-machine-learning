#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<math.h>

double* get_test_prediction(double **train, double **test, int column, int min_size, int max_depth, int fold_size, int train_size)
{
	double *predictions = (double *)malloc(fold_size * sizeof(double)); //预测集的行数就是数组prediction的长度
	struct treeBranch *tree = build_tree(train_size, column, train, min_size, max_depth);
	for (int i = 0; i < fold_size; i++)
	{
		 predictions[i] = predict(test[i], tree);
	}
	return predictions; //返回对test的预测数组
}