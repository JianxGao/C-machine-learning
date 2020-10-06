
//功能——归一化数据
//输入数据集的二维数组，行数，列数 
//返回归一化后的数据集二维数组

void normalize_dataset(float **dataset,int row, int col) 
{
	// 先 对列循环
	float maximum, minimum;
	for (int i = 0; i < col; i++) 
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
