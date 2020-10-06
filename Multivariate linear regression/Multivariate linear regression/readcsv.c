#include <stdio.h>
#include<stdlib.h>
#include <string.h>
#define row 4898
#define column 12

int readcsv(char* filename,int beginning, float data[row][column])
{
	FILE *fp = NULL;
	char* line;
	char* record;
	char buffer[1024];
	int i, j;
	if ((fp = fopen(filename, "at+")) != NULL)
	{
		fseek(fp, beginning, SEEK_SET);
		char delims[] = ",";
		int k = 0;
		int l = 0;
		while ((line = fgets(buffer, sizeof(buffer), fp)) != NULL)//当没有读取到文件末尾时循环继续,fgets返回值是缓冲区地址，buffer是缓冲区地址,读取sizeof(buffer)有可能不到一行，也有可能超过一行，超过一行则只读一行，所以在定义buffer时注意保证其大小超过一行
		{
			record = strtok(line, ",");//返回被分割片段的第一个字符的地址
			while (record != NULL)//读取每一行的数据
			{
				//printf("%s  ", record);//将读取到的每一个数据打印出来，输出字符串时，看到%s，printf就要求指针变量，而不是我们逻辑上认为的指针里面所存储的内容，必须提供字符串首地址！！！如果用%c，想输出里面的内容就可以按照正常的指针概念，用*p了
				data[k][l] = atof(record);
				//printf("datastr[%d][%d]=%f\n",k,l, data[k][l]);			
				record = strtok(NULL, ",");
				l += 1;
			}
			//printf("\n");
			l = 0;
			k += 1;
		}
		fclose(fp);
		/*for (i = 0; i < row; i++) {
			for (j = 0; j < column; j++) {
				printf("i=%d,j=%d,datastr=%f\n", i, j, data[i][j]);
			}
		}*/
	}
	return 0;
}