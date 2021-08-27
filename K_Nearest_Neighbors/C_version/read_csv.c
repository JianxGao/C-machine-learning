#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double **dataset;
int row, col;
int get_row(char *filename) //��ȡ����
{
	char line[1024];
	int i = 0;
	FILE *stream = fopen(filename, "r");
	while (fgets(line, 1024, stream))
	{
		i++;
	}
	fclose(stream);
	return i;
}

int get_col(char *filename) //��ȡ����
{
	char line[1024];
	int i = 0;
	FILE *stream = fopen(filename, "r");
	fgets(line, 1024, stream);
	char *token = strtok(line, ",");
	while (token)
	{
		token = strtok(NULL, ",");
		i++;
	}
	fclose(stream);
	return i;
}

void get_two_dimension(char *line, double **dataset, char *filename)
{
	FILE *stream = fopen(filename, "r");
	int i = 0;
	while (fgets(line, 1024, stream)) //���ж�ȡ
	{
		int j = 0;
		char *tok;
		char *tmp = strdup(line);
		for (tok = strtok(line, ","); tok && *tok; j++, tok = strtok(NULL, ",\n"))
		{
			dataset[i][j] = atof(tok); //ת���ɸ�����
		}							   //�ַ�����ֲ���
		i++;
		free(tmp);
	}
	fclose(stream); //�ļ��򿪺�Ҫ���йرղ���
}