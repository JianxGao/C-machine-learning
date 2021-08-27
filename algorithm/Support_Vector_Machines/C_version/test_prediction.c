#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

double *get_label(double **dataset, int row, int col)
{
	double *label = (double *)malloc(row * sizeof(double));
	for (int i = 0; i < row; i++)
	{
		label[i] = dataset[i][col - 1];
	}
	return label;
}

double array_dot(double *row1, double *row2, int col)
{
	double res = 0;
	for (int i = 0; i < col; i++)
	{
		res += row1[i] * row2[i];
	}
	return res;
}

double calculate_error(double b, double *label, double **data, double *alpha, int row, int column, int index)
{
	double error = 0;
	double *dot_res;
	dot_res = (double *)malloc(row * sizeof(double));
	for (int i = 0; i < row; i++)
	{
		dot_res[i] = array_dot(data[i], data[index], column - 1);

		dot_res[i] = dot_res[i] * alpha[i] * label[i];
		error += dot_res[i];
	}

	error += b - label[index];

	return error;
}

void Svm_Smo(double *b, double *alpha, int m_passes, double **train_data, double *label, double tol, double C, double change_limit, int row, int col)
{
	srand((unsigned)time(NULL));
	int p_num = 0;
	while (p_num < m_passes)
	{
		int num_chaged_alpha = 0;
		for (int i = 0; i < row; i++)
		{
			double error_i = calculate_error(*b, label, train_data, alpha, row, col, i);
			if (((label[i] * error_i < (-tol)) && (alpha[i] < C)) || ((label[i] * error_i > tol) && (alpha[i] > 0)))
			{
				int j = rand() % row;
				while (j == i)
				{
					j = rand() % row;
				}
				double error_j = calculate_error(*b, label, train_data, alpha, row, col, j);
				// save old alpha i, j
				double alpha_old_i = alpha[i];
				double alpha_old_j = alpha[j];
				// compute Land H
				double L = 0, H = C;
				if (label[i] != label[j])
				{
					double L = 0 > (alpha[j] - alpha[i]) ? 0 : (alpha[j] - alpha[i]);
					double H = C < (C + alpha[j] - alpha[i]) ? C : (C + alpha[j] - alpha[i]);
				}
				else
				{
					double L = 0 > (alpha[j] + alpha[i] - C) ? 0 : (alpha[j] + alpha[i] - C);
					double H = C < (alpha[j] + alpha[i]) ? C : (alpha[j] + alpha[i]);
				}
				if (L == H)
				{
					continue;
				}
				// compute eta, in order to be convenient to judge
				double eta = 2 * array_dot(train_data[i], train_data[j], col - 1) -
							 array_dot(train_data[i], train_data[i], col - 1) -
							 array_dot(train_data[j], train_data[j], col - 1);
				if (eta >= 0)
				{
					continue;
				}
				// computeand clip new value for alpha_raw_j
				alpha[j] -= (label[j] * (error_i - error_j) / eta);
				// compute alpha_new_j
				if (alpha[j] > H)
				{
					alpha[j] = H;
				}
				else if (alpha[j] < L)
				{
					alpha[j] = L;
				}
				// Check
				if (fabs(alpha[j] - alpha_old_j) < change_limit)
				{
					continue;
				}
				// compute alpha_new_i
				alpha[i] += label[i] * label[j] * (alpha_old_j - alpha[j]);
				// compute b1, b2
				double b1 = *b - error_i - label[i] * (alpha[i] - alpha_old_i) * array_dot(train_data[i], train_data[i], col - 1) -
							label[j] * (alpha[j] - alpha_old_j) * array_dot(train_data[i], train_data[j], col - 1);
				double b2 = *b - error_j - label[i] * (alpha[i] - alpha_old_i) * array_dot(train_data[i], train_data[j], col - 1) -
							label[j] * (alpha[j] - alpha_old_j) * array_dot(train_data[j], train_data[j], col - 1);
				if ((0 < alpha[i]) && (alpha[i] < C))
				{
					*b = b1;
				}
				else if ((0 < alpha[j]) && (alpha[j] < C))
				{
					*b = b2;
				}
				else
				{
					*b = (b1 + b2) / 2;
					num_chaged_alpha += 1;
				}
			}
			else
			{
				continue;
			}
		}
		if (num_chaged_alpha == 0)
		{
			p_num += 1;
		}
		else
		{
			p_num = 0;
		}
	}
}

double *get_weight(double *alpha, double *label, double **train_data, int row, int col)
{
	double *weight;
	weight = (double *)malloc((col - 1) * sizeof(double));
	for (int j = 0; j < col - 1; j++)
	{
		weight[j] = 0;
		for (int i = 0; i < row; i++)
		{
			weight[j] += alpha[i] * label[i] * train_data[i][j];
		}
	}
	return weight;
}

double predict(double *w, double *test_data, double b, int col)
{
	return array_dot(test_data, w, col - 1) + b;
}

double *get_test_prediction(double **train, int train_size, double **test_data, int test_size, int m_passes, double tol, double C, double change_limit, int col)
{
	double b = 0;
	double *alpha = (double *)malloc(train_size * sizeof(double));
	for (int i = 0; i < train_size; i++)
	{
		alpha[i] = 0;
	}
	double *label = get_label(train, train_size, col);
	Svm_Smo(&b, alpha, m_passes, train, label, tol, C, change_limit, train_size, col);
	double *w = get_weight(alpha, label, train, train_size, col);
	double *predictions = (double *)malloc(test_size * sizeof(double));
	for (int i = 0; i < test_size; i++)
	{
		predictions[i] = predict(w, test_data[i], b, col);
		if (predictions[i] >= 0)
		{
			predictions[i] = 1;
		}
		else
		{
			predictions[i] = -1;
		}
	}
	return predictions;
}