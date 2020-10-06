#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

//#include <iostream>
//#include <vector>
//#include <string>
//#include <fstream>
//#include <sstream>

#define OUT_COUT  3     //输出向量维数
#define IN_COUT   7     //输入向量维数
#define COUT      150    //训练集数量，数据已经标准化 
#define TEST_COUNT  50    //测试集数量，数据已经标准化 

typedef struct {       //bp人工神经网络结构
    int h;             //实际使用隐层数量
    double v[IN_COUT][50];   //隐藏层权矩阵i,隐层节点最大数量为100
    double w[50][OUT_COUT];   //输出层权矩阵
    double a;          //学习率
    double b;          //精度控制参数
    int LoopCout;      //最大循环次数
} bp_nn;


//*******************************************************************************************************************************************

//Sigmoid函数
double fnet(double net) { 
    return 1/(1+exp(-net));
}

//初始化bp网络
int InitBp(bp_nn *bp) { 
   
    printf("请输入隐层节点数，最大数为100（题设为5）：\n");   
    scanf("%d", &(*bp).h);
   
    printf("请输入学习率（题设为0.3）：\n");
    scanf("%lf", &(*bp).a);    

    printf("请输入精度控制参数：\n");
    scanf("%lf", &(*bp).b);

    printf("请输入最大循环次数（题设为500）：\n");
    scanf("%d", &(*bp).LoopCout);


	//**初始化网络(1)
	// **
    int i, j;
    srand((unsigned)time(NULL));
    for (i = 0; i < IN_COUT; i++)
        for (j = 0; j < (*bp).h; j++)
            (*bp).v[i][j] = rand() / (double)(RAND_MAX);   //随机化权重矩阵 
    for (i = 0; i < (*bp).h; i++)
        for (j = 0; j < OUT_COUT; j++)
            (*bp).w[i][j] = rand() / (double)(RAND_MAX);   //随机化权重矩阵 
   //**
   
   
    return 1;
}

//训练bp网络，样本为x，理想输出为y

//*训练网络（整个叫训练网络trainBP(5)   ，里面包括了 初始化网络 计算活跃度 前向传播 计算反向传播误差 更新权重 
//** 
int TrainBp(bp_nn *bp, double x[COUT][IN_COUT], int y[COUT][OUT_COUT]) {
    double f = (*bp).b;                      //精度控制参数
    double a = (*bp).a;                      //学习率
    int h = (*bp).h;                         //隐层节点数
    double v[IN_COUT][50], w[50][OUT_COUT]; //权重矩阵
    double Ch_v[IN_COUT][50], Ch_w[50][OUT_COUT]; //权重矩阵修改量
    double ChgH[50], ChgO[OUT_COUT];         //修改量矩阵
    double O1[50], O2[OUT_COUT];             //隐层和输出层输出量
    int LoopCout = (*bp).LoopCout;           //最大循环次数
    int i, j, k, n;
    double temp;

    for (i = 0; i < IN_COUT; i++)            // weight
        for (j = 0; j < h; j++)
            v[i][j] = (*bp).v[i][j];
    for (i = 0; i < h; i++)
        for (j = 0; j < OUT_COUT; j++)
            w[i][j] = (*bp).w[i][j];
   
    double e = f + 1;
    for (n = 0; e > f && n < LoopCout; n++) { //对每个样本训练网络
        
		e = 0;
        for (j = 0; j < OUT_COUT; j++)
            ChgO[j] = 0;
        for (j = 0; j < h; j++)
            ChgH[j] = 0;
        for (j = 0; j < h; j++)         
            for (k = 0; k < OUT_COUT; k++)
                Ch_w[j][k] = 0;
        for (j = 0; j < IN_COUT; j++)   
            for (k = 0; k < h; k++)
                Ch_v[j][k] = 0;
        
		for (i= 0; i < COUT; i++) {
            
            
            //*计算活跃度activate(2)
			//** 
			for (k= 0; k < h; k++) {          //计算隐层输出向量
                temp = 0;
                for (j = 0; j < IN_COUT; j++)
                    temp = temp + x[i][j] * v[j][k];   
                O1[k] = fnet(temp);
            }
            //**
            
            //*前向传播forward_propagate(3) 
            //**
            for (k = 0; k < OUT_COUT; k++) { //计算输出层输出向量
                temp = 0;
                for (j = 0; j < h; j++)
                    temp = temp + O1[j] * w[j][k];
                O2[k] = fnet(temp);
            }
            //**
            
            //*计算反向传播误差backward_propagate_error(3)
            //**
            for (j = 0; j < OUT_COUT ; j++)   //计算输出误差
                e = e + (y[i][j] - O2[j]) * (y[i][j] - O2[j]);
            for (j = 0; j < OUT_COUT; j++)   
                ChgO[j] = O2[j] * (1 - O2[j]) * (y[i][j] - O2[j]);
            for (j = 0; j < h; j++)         
                for (k = 0; k < OUT_COUT; k++)
                    Ch_w[j][k] += a * O1[j] * ChgO[k]; //累加所有样本训练后的改变量
            for (j = 0; j < h; j++) {     
                temp = 0;
                for (k = 0; k < OUT_COUT; k++)
                    temp = temp + w[j][k] * ChgO[k];
                ChgH[j] = temp * O1[j] * (1 - O1[j]);
            }
            for (j = 0; j < IN_COUT; j++)   
                for (k = 0; k < h; k++)
                    Ch_v[j][k] += a * x[i][j] * ChgH[k]; //累加所有样本训练后的改变量，消除样本顺序影响
            //**
            
        }
        
        //*更新权重update_weights(4)
        //**
        for (j = 0; j < h; j++)           //修改输出层权矩阵
            for (k = 0; k < OUT_COUT; k++)
                w[j][k] = w[j][k] + Ch_w[j][k];
        for (j = 0; j < IN_COUT; j++)     //修改隐藏层权矩阵
            for (k = 0; k < h; k++)
                v[j][k] = v[j][k] + Ch_v[j][k];
        //**
    }
    printf("总共循环次数：%d\n", n);
    printf("调整后的隐层权矩阵：\n");
    for (i = 0; i < IN_COUT; i++) {   
        for (j = 0; j < h; j++)
            printf("%f    ", v[i][j]);   
        printf("\n");
    }
    printf("调整后的输出层权矩阵：\n");
    for (i = 0; i < h; i++) {
        for (j = 0; j < OUT_COUT; j++)
            printf("%f    ", w[i][j]);   
        printf("\n");
    }
    for (i = 0; i < IN_COUT; i++)             //把结果复制回结构体
        for (j = 0; j < h; j++)
            (*bp).v[i][j] = v[i][j];
    for (i = 0; i < h; i++)
        for (j = 0; j < OUT_COUT; j++)
            (*bp).w[i][j] = w[i][j];
    printf("BP网络训练结束！\n");

    return 1;
}
//** 

//使用bp网络useBP(6)

//*由训练所得网络计算预测值
//**
int UseBp(bp_nn *bp) {    
    double Input[TEST_COUNT][IN_COUT];
    int Output[TEST_COUNT][1];
    int count=0;
    int flag = 0;
    
	double O1[50];
    double O2[OUT_COUT]; //O1为隐层输出,O2为输出层输出
    int num=0;
    
	ReadFromCSV("test_data.csv",Input,num);    //读文件（x） 
	ReadFromCSV1("test_data0.csv",Output,num); //读文件（理想输出y） 
	
    for(int a=0;a<TEST_COUNT;a++) {              
		int i, j;
	    double temp;
        
		for (i = 0; i < (*bp).h; i++) {
            temp = 0;
            for (j = 0; j < IN_COUT; j++)
                temp += Input[a][j] * (*bp).v[j][i];
            O1[i] = fnet(temp);
        }
        for (i = 0; i < OUT_COUT; i++) {
            temp = 0;
            for (j = 0; j < (*bp).h; j++)
                temp += O1[j] * (*bp).w[j][i];
            O2[i] = fnet(temp);
        }
        //printf("数据%.4f  %.4f %.4f\n",Input[a][0],Input[a][1],Input[a][2]);
        printf("结果：   ");
            printf("%.3f  %.3f %.3f",O2[0],O2[1],O2[2]);    //三个输出 
        printf("\n");
        
        //进行分类 
		if((O2[1]-O2[0])>0.0001) {
			if((O2[2]-O2[1]>0.0001)){
				flag = 3; 
				printf("第三类\n\n");
			}
			else{
				flag = 2;
				printf("第二类\n\n");
			} 
		}else{
			flag = 1;
			printf("第一类\n\n");
		}
		
		//计算准确率 
		if(flag == Output[a][1]) count++;
    }
    printf("Accuracy Rate %.4f\n",double(count)/double(TEST_COUNT));
    return 1;
    
}

//**


//
double x[COUT][IN_COUT];
int y[COUT][OUT_COUT];
int main(int argc, char const *argv[])
{
    
	int num=0;
	ReadFromCSV("training_data.csv",x,num);   //读文件（x） 

	int num0=0;
	ReadFromCSV0("training_data0.csv",y,num0); //读文件（理想输出y） 

    bp_nn bp;

    InitBp(&bp);                    //初始化bp网络结构
    TrainBp(&bp, x, y);             //训练bp神经网络
    UseBp(&bp);                     //测试bp神经网络
	
    return 1;
} 

//////////////////////////////////////////// main函数调用格式 /////////////////////////////////////////////////////////////
extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern void normalize_dataset(float **dataset,int row, int col);
extern float rmse_metric(float *actual, float *predicted, int fold_size);
extern float accuracy_metric(float *actual, float *predicted, int fold_size);
extern double  cross_validation_split(double **dataset, int row, int n_folds, int fold_size);
extern float evaluate_algorithm(float **dataset, int n_folds, int fold_size, float l_rate, int n_epoch) ;
extern float get_test_prediction(float **train, float **test, float l_rate, int n_epoch, int fold_size);


int main()
{
	//读取csv 
	char filename[] = "seeds_data.csv";
    char line[1024];
    row = get_row(filename);
    col = get_col(filename);
    dataset = (double **)malloc(row*sizeof(int *));
	for (int i = 0; i < row; ++i){
		dataset[i] = (double *)malloc(col*sizeof(double));
	}//动态申请二维数组	
	get_two_dimension(line, dataset, filename);
	
	//设置相应参数 
	float l_rate = 0.1;		//学习率 
	int epoch = 200;		//循环次数 
	int n_folds = 5;		//数据分组数 
    fold_size=(int)(row/n_folds);		//每组的数据量 
    
	//归一化数据
	normalize_dataset(dataset, row, col) 
	
	//将数据划分为 k 组
	cross_validation_split(dataset, row, n_folds, fold_size)	
	 
    //按划分的k折交叉验证计算预测所得准确率
    evaluate_algorithm(dataset, n_folds, fold_size, l_rate, n_epoch) 
    
    
}
 
