#include "BA.h"

// 切分函数，根据切分点将数据分为左右两组
struct dataset *test_split(int index, double value, int row, int col, double **data)
{
	// 将切分结果作为结构体返回
	struct dataset *split = (struct dataset *)malloc(sizeof(struct dataset));
	int count1=0,count2=0;
	double ***groups = (double ***)malloc(2 * sizeof(double **));
	for (int i = 0; i < 2; i++)
	{
		groups[i]=(double **)malloc(row * sizeof(double *));
		for (int j = 0; j < row; j++)
		{
			groups[i][j] = (double *)malloc(col * sizeof(double ));
		}
	}
	for (int i = 0; i < row; i++)
	{
		if (data[i][index]<value)
		{
			groups[0][count1]=data[i];
			count1 ++;
		}else{
			groups[1][count2] = data[i];
			count2++;
		}
	}
	split->splitdata = groups;
	split->row1 = count1;
	split->row2 = count2;
	return split;
}

// 计算Gini系数
double gini_index(int index,double value,int row, int col, double **dataset, double *class, int classnum)
{
	float *numcount1 = (float *)malloc(classnum * sizeof(float));
	float *numcount2 = (float *)malloc(classnum * sizeof(float));
	for (int i = 0; i < classnum; i++)
		numcount1[i]=numcount2[i]=0;

	float count1 = 0, count2 = 0;
	double gini1,gini2,gini;
	gini1=gini2=gini=0;
	// 计算每一类的个数
	for (int i = 0; i < row; i++)
	{
		if (dataset[i][index] < value)
		{
			count1 ++;
			for (int j = 0; j < classnum; j++)
				if (dataset[i][col-1]==class[j])
					numcount1[j] += 1;
		}
		else
		{
			count2++;
			for (int j = 0; j < classnum; j++)
				if (dataset[i][col - 1] == class[j])
					numcount2[j]++;
		}
	}
	// 判断分母是否为0，防止运算错误
	if (count1==0)
	{
		gini1=1;
		for (int i = 0; i < classnum; i++)
			gini2 += (numcount2[i] / count2) * (numcount2[i] / count2);
	}else if (count2==0)
	{
		gini2=1;
		for (int i = 0; i < classnum; i++)
			gini1 += (numcount1[i] / count1) * (numcount1[i] / count1);
	}else
	{
		for (int i = 0; i < classnum; i++)
		{
			gini1 += (numcount1[i] / count1) * (numcount1[i] / count1);
			gini2 += (numcount2[i] / count2) * (numcount2[i] / count2);
		}
	}
	// 计算Gini系数
	gini1 = 1 - gini1;
	gini2 = 1 - gini2;
	gini = (count1 / row) * gini1 + (count2 / row) * gini2;
	free(numcount1);free(numcount2);
	numcount1=numcount2=NULL;
	return gini;
}

// 选取数据的最优切分点
struct treeBranch *get_split(int row, int col, double **dataset, double *class, int classnum, int n_features)
{
	struct treeBranch *tree=(struct treeBranch *)malloc(sizeof(struct treeBranch));
	int *featurelist=(int *)malloc(n_features * sizeof(int));
	int count=0,flag=0,temp;
	int b_index=999;
	double b_score = 999, b_value = 999,score;
	// 随机选取n_features个特征
	while (count<n_features)
	{
		featurelist[count]=count;
		count++;
	}
	// 计算所有切分点Gini系数，选出Gini系数最小的切分点
	for (int i = 0; i < n_features; i++)
	{
		for (int j = 0; j < row; j++)
		{
			double value=dataset[j][featurelist[i]];
			score = gini_index(featurelist[i], value, row, col, dataset, class, classnum);
			if (score<b_score)
			{
				b_score = score;
				b_value = value;
				b_index = featurelist[i];
			}
		}
	}
	tree->index=b_index;tree->value=b_value;tree->flag=0;
	return tree;
}

// 计算叶节点结果
double to_terminal(int row, int col, double **data, double *class, int classnum)
{
	int *num=(int *)malloc(classnum*sizeof(classnum));
	double maxnum=0;
	int flag=0;
	// 计算所有样本中结果最多的一类
	for (int i = 0; i < classnum; i++)
		num[i]=0;
	for (int i = 0; i < row; i++)
		for (int j = 0; j < classnum; j++)
			if (data[i][col-1]==class[j])
				num[j]++;
	for (int j = 0; j < classnum; j++)
	{
		if (num[j] > flag)
		{
			flag = num[j];
			maxnum = class[j];
		}
	}
	free(num);
	num=NULL;
	return maxnum;
}

// 创建子树或生成叶节点
void split(struct treeBranch *tree, int row, int col, double **data, double *class, int classnum, int depth, int min_size, int max_depth, int n_features)
{
	// 判断是否已经达到最大层数
	if (depth>=max_depth)
	{
		tree->flag=1;
		tree->output = to_terminal(row, col, data, class, classnum);
		return;
	}
	struct dataset *childdata = test_split(tree->index, tree->value, row, col, data);
	// 判断样本是否已被分为一边
	if (childdata->row1==0 || childdata->row2==0)
	{
		tree->flag = 1;
		tree->output = to_terminal(row, col, data, class, classnum);
		return;
	}
	// 左子树，判断样本是否达到最小样本数，如不是则继续迭代
	if (childdata->row1<=min_size)
	{
		struct treeBranch *leftchild = (struct treeBranch *)malloc(sizeof(struct treeBranch));
		leftchild->flag=1;
		leftchild->output = to_terminal(childdata->row1, col, childdata->splitdata[0], class, classnum);
		tree->leftBranch=leftchild;
	}
	else
	{
		struct treeBranch *leftchild = get_split(childdata->row1, col, childdata->splitdata[0], class, classnum, n_features);
		tree->leftBranch=leftchild;
		split(leftchild, childdata->row1, col, childdata->splitdata[0], class, classnum, depth + 1, min_size, max_depth, n_features);
	}
	// 右子树，判断样本是否达到最小样本数，如不是则继续迭代
	if (childdata->row2 <= min_size)
	{
		struct treeBranch *rightchild = (struct treeBranch *)malloc(sizeof(struct treeBranch));
		rightchild->flag = 1;
		rightchild->output = to_terminal(childdata->row2, col, childdata->splitdata[1], class, classnum);
		tree->rightBranch = rightchild;
	}
	else
	{
		struct treeBranch *rightchild = get_split(childdata->row2, col, childdata->splitdata[1], class, classnum, n_features);
		tree->rightBranch = rightchild;
		split(rightchild, childdata->row2, col, childdata->splitdata[1], class, classnum, depth + 1, min_size, max_depth, n_features);
	}
	free(childdata->splitdata);childdata->splitdata=NULL;
	free(childdata);childdata=NULL;
	return;
}

// 生成决策树
struct treeBranch *build_tree(int row, int col, double **data, int min_size, int max_depth, int n_features)
{
	int count1 = 0, flag1 = 0;
	// 判断结果一共有多少类别，此处classes[20]仅仅是取一个较大的数20，默认类别不可能超过20类
	double classes[20];
	for (int i = 0; i < row; i++)
	{
		if (count1 == 0)
		{
			classes[0] = data[i][col - 1];
			count1++;
		}
		else
		{
			flag1 = 0;
			for (int j = 0; j < count1; j++)
				if (classes[j] == data[i][col - 1])
					flag1 = 1;
			if (flag1 == 0)
			{
				classes[count1] = data[i][col - 1];
				count1++;
			}
		}
	}
	// 生成切分点
	struct treeBranch *result = get_split(row, col, data, classes, count1, n_features);
	// 进入迭代，不断生成子树
	split(result, row, col, data, classes, count1, 1, min_size, max_depth, n_features);
	return result;
}

// 随机森林算法，将森林结果保存为结构体数组，并返回结构体二重指针
struct treeBranch **random_forest(int row, int col, double **data, int min_size, int max_depth, int n_features, int n_trees, float sample_size)
{
	struct treeBranch ** forest = (struct treeBranch **)malloc(n_trees * sizeof(struct treeBranch*));
	int samplenum = (int)(row*sample_size);
	int temp;
	// 生成随机训练集
	double ** subsample=(double **)malloc(samplenum * sizeof(double *));
	for (int i = 0; i < samplenum; i++)
	{
		subsample[i]=(double *)malloc(col * sizeof(double));
	}
	// 生成所有决策树
	for (int j = 0; j < n_trees; j++)
	{
		for (int i = 0; i < samplenum; i++)
		{
			temp = rand() % row;
			subsample[i] = data[temp];
		}
		struct treeBranch *tree = build_tree(samplenum, col, subsample, min_size, max_depth, n_features);
		forest[j]=tree;
	}
	return forest;
}

// 决策树预测
double treepredict(double *test, struct treeBranch *tree)
{
	double output;
	// 判断是否达到叶节点，flag=1时为叶节点，flag=0时则继续判断
	if (tree->flag==1)
	{
		output = tree->output;
		return output;
	}
	else
	{
		if (test[tree->index]<tree->value)
		{
			output = treepredict(test, tree->leftBranch);
			return output;
		}
		else
		{
			output = treepredict(test, tree->rightBranch);
			return output;
		} 
	}
}

// 随机森林bagging预测
double predict(double *test, struct treeBranch **forest, int n_trees)
{
	double output;
	double * forest_result=(double *)malloc(n_trees * sizeof(double));
	double * classes = (double *)malloc(n_trees * sizeof(double));
	int * num = (int *)malloc(n_trees * sizeof(int));
	for (int i = 0; i < n_trees; i++)
		num[i]=0;
	int count=0,flag,temp=0;
	// 将每棵树的判断结果保存
	for (int i = 0; i < n_trees; i++)
		forest_result[i] = treepredict(test, forest[i]);
	// bagging选出最后结果
	for (int i = 0; i < n_trees; i++)
	{
		flag=0;
		for (int j = 0; i < count; i++)
		{
			if (forest_result[i]==classes[j])
			{
				flag=1;
				num[j]++;
			}
		}
		if (flag==0)
		{
			classes[count]=forest_result[i];
			num[count]++;
			count++;
		}
	}
	for (int i = 0; i < count; i++)
	{
		if (num[i]>temp)
		{
			temp=num[i];
			output=classes[i];
		}
	}
	return output;
}

// 从样本集中进行有放回子采样得到子样本集
double **subsample(double **dataset, double ratio)
{
	int n_sample = (int)(row * ratio + 0.5);
	double **sample=0;
	sample = (double **)malloc(n_sample*sizeof(int *));
	for (int i = 0; i < n_sample; i++){
		sample[i] = (double *)malloc(col*sizeof(double));
	}
	int i,j,Index;
	for(i=0;i<n_sample;i++)
	{
		Index = rand() % row;
		for(j=0;j<col;j++)
		{
			sample[i][j] = dataset[Index][j];
		}
	}
	return sample;
}

// 用每个基模型进行预测并投票决定最终预测结果
double bagging_predict(struct treeBranch **trees, double *row, int n_trees)
{
	int i;
	int n_classes = 1000;
	int class_count[1000] = {0};
	double *predictions=0;
	predictions = (double *)malloc(n_trees*sizeof(double));
	double counts = 0;
	double result = 0;
	for(i=0;i<n_trees;i++)
	{
		predictions[i] = predict(row,&trees[i],n_trees);
		class_count[(int)predictions[i]] += 1;
	}
	for(i=0;i<n_classes;i++)
	{
		if(class_count[i]>counts)
		{
			counts = class_count[i];
			result = i;
		}
	}
	return result;
}

// Bagging 算法实现
double bagging(double **train, double **test, int max_depth, int min_size, double sample_size, int n_trees, int n_features)
{
	int i,j;
	int n_classes = 1000;
	int class_count[1000] = {0};
	double **predictions = 0;
	predictions = (double **)malloc(n_trees*sizeof(int));
	for (int i = 0; i < n_trees; i++){
		predictions[i] = (double *)malloc(sizeof(test)/sizeof(test[0])*sizeof(double));
	}
	double counts = 0;
	double result = 0;
	double *results = 0;
	results = (double *)malloc(sizeof(test)/sizeof(test[0])*sizeof(double *));
	for(i=0;i<n_trees;i++)
	{
		double **sample = subsample(train,sample_size);  // sample_size表示ratio
		struct treeBranch *tree = build_tree(row, col, sample, min_size, max_depth, n_features);
		for(j=0;j<sizeof(test)/sizeof(test[0]);j++)
		{
			predictions[i][j] = predict(test[j],&tree,n_features);
		}
	}
	for(j=0;j<sizeof(test)/sizeof(test[0]);j++)
	{
		for(i=0;i<n_trees;i++)
		{
			class_count[(int)predictions[i][j]] += 1;
		}
		for(i=0;i<n_classes;i++)
		{
			if(class_count[i]>counts)
			{
				counts = class_count[i];
				result = i;
			}
		}
		results[j] = result;
	}
	return *results;
}