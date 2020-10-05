#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void normalize_dataset(double **dataset, int row, int col)
{
	// 先 对列循环
	double maximum, minimum;
	for (int i = 0; i < col - 1; i++)
	{
		// 第一行为标题，值为0，不能参与计算最大最小值
		maximum = dataset[0][i];
		minimum = dataset[0][i];
		//再 对行循环
		for (int j = 0; j < row; j++)
		{
			maximum = max(dataset[j][i], maximum);
			minimum = min(dataset[j][i], minimum);
		}
		// 归一化处理
		for (int j = 0; j < row; j++)
		{
			dataset[j][i] = (dataset[j][i] - minimum) / (maximum - minimum);
		}
	}
}