#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

int get_row(char *filename)//获取行数
{
	char line[1024];
	int i = 0;
	FILE* stream = fopen(filename, "r");
	while (fgets(line, 1024, stream)) {
		i++;
	}
	fclose(stream);
	return i;
}

int get_col(char *filename)//获取列数
{
	char line[1024];
	int i = 0;
	FILE* stream = fopen(filename, "r");
	fgets(line, 1024, stream);
	char* token = strtok(line, ",");
	while (token) {
		token = strtok(NULL, ",");
		i++;
	}
	fclose(stream);
	return i;
}

void get_two_dimension(char *line, double **dataset, char *filename)
{
	FILE* stream = fopen(filename, "r");
	int i = 0;
	while (fgets(line, 1024, stream))//逐行读取
	{
		int j = 0;
		char *tok;
		char *tmp = _strdup(line);
		for (tok = strtok(line, ","); tok && *tok; j++, tok = strtok(NULL, ",\n")) {
			dataset[i][j] = atof(tok);//转换成浮点数
		}//字符串拆分操作
		i++;
		free(tmp);
	}
	fclose(stream);//文件打开后要进行关闭操作
}