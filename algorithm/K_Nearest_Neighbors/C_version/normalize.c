#include <stdlib.h>
#include <stdio.h>

void normalize_dataset(double **dataset, int row, int col)
{
	// �� ����ѭ��
	double maximum, minimum;
	for (int i = 0; i < col; i++)
	{
		// ��һ��Ϊ���⣬ֵΪ0�����ܲ�����������Сֵ
		maximum = dataset[0][i];
		minimum = dataset[0][i];
		//�� ����ѭ��
		for (int j = 0; j < row; j++)
		{
			maximum = (dataset[j][i] > maximum) ? dataset[j][i] : maximum;
			minimum = (dataset[j][i] < minimum) ? dataset[j][i] : minimum;
		}
		// ��һ������
		for (int j = 0; j < row; j++)
		{
			dataset[j][i] = (dataset[j][i] - minimum) / (maximum - minimum);
		}
	}
}