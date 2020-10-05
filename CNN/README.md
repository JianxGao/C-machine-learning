## CNN

### 1.算法介绍

#### 	1.1  DNN（全连接）

​	全连接深度神经网络，顾名思义，每个神经元都与相邻层的神经元连接。在这个实验中，每个数字的image是28*28，也就是784(=28*28)个数值，每个数值对应一个像素值，值的大小反应像素点的强度。这就意味着我们网络的输入层有784个神经元。输出层呢？由于我们是预测0-9这几个数字，输出层当然就是10个神经元了。至于隐藏层节点的个数我们可以自行选定。

![tikz41](.\pic\tikz41.png)

​	如图为输入层、三个隐藏全连接层、一个输出层。

#### 1.2 CNN三大思想

##### 		1.2.1 局部感受野

​			刚刚我们在DNN中是把输入层784个神经元排成了一条长线，这里我们还原图片原来的样子(28*28)，如下图

![tikz42](.\pic\tikz42.png)

​			DNN中，我们会把输入层的每个神经元都与第一个隐藏层的每个神经元连接。而在CNN中我们这样做的，第一个隐藏层的神经元只与局部区域输入层的神经元相连。下图就是第一个隐藏层的某个神经元与局部区域输入层的神经元相连的情况。

![tikz43](.\pic\tikz43.png)

​			这里的局部区域就是局部感受野，它像一个架在输入层上的窗口。你可以认为某一个隐藏层的神经元学习分析了它”视野范围“(局部感受野)里的特征。图中一个隐藏层的神经元有5*5个权值参数与之对应。移动这样一个窗口使它能够扫描整张图，每次移动它都会有一个不同的节点与之对应。

![tikz44](.\pic\tikz44.png)

​			            ![tikz45](.\pic\tikz45.png)

​			以此类推可以形成第一个隐藏层，注意我们的图片是28*28的，窗口是5*5的，可以得到一个24*24(24=28-5+1)个神经元的隐藏层。这里我们的窗口指滑动了一个像素，通常说成一步(stride)，也可以滑动多步，这里的stride也是一个超参，训练是可以根据效果调整，同样，窗口大小也是一个超参。

##### 		1.2.2权值共享

​			权值和偏移值是共享公式如下：

​                                                     $$\sigma \left(b+\sum_){l=0}^{4}\sum_{m=0}^{4}{w_{l,m}a_{j+l,k+m}} \right )$$

​			σ代表的是激活函数，如sigmoid函数等，b就是偏移值，w就是5*5个共享权值矩阵，我们用矩阵a表示输入层的神经元，ax,y表示第x+1行第y+1列那个神经元(注意，这里的下标默认都是从0开始计的，a0,0表示第一行第一列那个神经元）所以通过矩阵w线性mapping后再加上偏移值就得到公式中括号里的式子，表示的是隐藏层中第j+1行k+1列那个神经元的输入。

​                                                                      $$a^1 = \sigma(b + w * a^0)$$

##### 		1.2.3池化（Pooling）

​			CNN还有一个重要思想就是池化，池化层通常接在卷积层后面。池化这个词听着就很有学问，其实引入它的目的就是为了简化卷积层的输出。通俗地理解，池化层也在卷积层上架了一个窗口，但这个窗口比卷积层的窗口简单许多，不需要w，b这些参数，它只是对窗口范围内的神经元做简单的操作，如求和，求最大值，把求得的值作为池化层神经元的输入值，如下图，这是一个2*2的窗口 。

![tikz47](.\pic\tikz47.png)

​			怎么理解max-pooling呢？由于经过了卷积操作，模型从输入层学到的特征反映在卷积层上，max-pooling做的事就是去检测这个特征是否在窗口覆盖范围的区域内。这也导致了，它会丢失这种特征所在的精准位置信息，所幸的是池化层可以保留相对位置信息。而后者相比而言比前者更重要。不理解上面的话也没关系，但是需要记住池化层一个最大的好处：经过池化后，大大减少了我们学到的特征值，也就大大减少了后面网络层的参数(上图可以看出池化层的神经元数明显少于卷积层神经元数)。

#### 	1.3总体结构

​		从左往右依次是输入层，卷积层，池化层，输出层。输入层到卷积层，卷积层到池化层已经详细介绍过了。池化层到输出层是全连接，这和DNN是一样的。

![tikz49](.\pic\tikz49.png)

​		

### 2.算法讲解

#### 			2.1  介绍

​		因此CNN代码使用的样本集是mnist手写字体识别集，格式与前的CSV文件不大相同，故使用不同的数据处理方式。

#### 			2.2  数据处理

使用数据集为http://yann.lecun.com/exdb/mnist/上的mnist集。

定义数据结构

```c
typedef struct MinstImg{
	int c;           // 图像宽
	int r;           // 图像高
	float** ImgData; // 图像数据二维动态数组
}MinstImg;

typedef struct MinstImgArr{
	int ImgNum;        // 存储图像的数目
	MinstImg* ImgPtr;  // 存储图像数组指针
}*ImgArr;              // 存储图像数据的数组

typedef struct MinstLabel{
	int l;            // 输出标记的长
	float* LabelData; // 输出标记数据
}MinstLabel;

typedef struct MinstLabelArr{
	int LabelNum;
	MinstLabel* LabelPtr;
}*LabelArr;              // 存储图像标记的数组
```

读取数据（返回图片数组、标签数组）

```c
ImgArr read_Img(const char* filename) // 读入图像
{
	FILE  *fp=NULL;
	fp=fopen(filename,"rb");
	if(fp==NULL)
		printf("open file failed\n");
	assert(fp);

	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;
	//从文件中读取sizeof(magic_number) 个字符到 &magic_number
	fread((char*)&magic_number,sizeof(magic_number),1,fp);
	magic_number = ReverseInt(magic_number);
	//获取训练或测试image的个数number_of_images
	fread((char*)&number_of_images,sizeof(number_of_images),1,fp);
	number_of_images = ReverseInt(number_of_images);
	//获取训练或测试图像的高度Heigh
	fread((char*)&n_rows,sizeof(n_rows),1,fp);
	n_rows = ReverseInt(n_rows);
	//获取训练或测试图像的宽度Width
	fread((char*)&n_cols,sizeof(n_cols),1,fp);
	n_cols = ReverseInt(n_cols);
	//获取第i幅图像，保存到vec中
    
    printf("幻数：%d\t数量:%d\t行:%d\t列:%d\t",magic_number,number_of_images,n_rows,n_cols);
	int i,r,c;

	// 图像数组的初始化
	ImgArr imgarr=(ImgArr)malloc(sizeof(MinstImg));
	imgarr->ImgNum=number_of_images;
	imgarr->ImgPtr=(MinstImg*)malloc(number_of_images*sizeof(MinstImg));

	for(i = 0; i < number_of_images; ++i)
	{
		imgarr->ImgPtr[i].r=n_rows;
		imgarr->ImgPtr[i].c=n_cols;
		imgarr->ImgPtr[i].ImgData=(float**)malloc(n_rows*sizeof(float*));
		for(r = 0; r < n_rows; ++r)
		{
			imgarr->ImgPtr[i].ImgData[r]=(float*)malloc(n_cols*sizeof(float));
			for(c = 0; c < n_cols; ++c)
			{
				unsigned char temp = 0;
				fread((char*) &temp, sizeof(temp),1,fp);
				imgarr->ImgPtr[i].ImgData[r][c]=(float)temp/255.0;
			}
		}
	}
	fclose(fp);
	return imgarr;
}
LabelArr read_Lable(const char* filename)// 读入图像
{
	FILE  *fp=NULL;
	fp=fopen(filename,"rb");
	if(fp==NULL)
		printf("open file failed\n");
	assert(fp);

	int magic_number = 0;
	int number_of_labels = 0;
	int label_long = 10;

	//从文件中读取sizeof(magic_number) 个字符到 &magic_number
	fread((char*)&magic_number,sizeof(magic_number),1,fp);
	magic_number = ReverseInt(magic_number);
	//获取训练或测试image的个数number_of_images
	fread((char*)&number_of_labels,sizeof(number_of_labels),1,fp);
	number_of_labels = ReverseInt(number_of_labels);

	int i,l;

	// 图像标记数组的初始化
	LabelArr labarr=(LabelArr)malloc(sizeof(MinstLabel));
	labarr->LabelNum=number_of_labels;
	labarr->LabelPtr=(MinstLabel*)malloc(number_of_labels*sizeof(MinstLabel));

	for(i = 0; i < number_of_labels; ++i)
	{
		labarr->LabelPtr[i].l=10;
		labarr->LabelPtr[i].LabelData=(float*)calloc(label_long,sizeof(float));
		unsigned char temp = 0;
		fread((char*) &temp, sizeof(temp),1,fp);
		labarr->LabelPtr[i].LabelData[(int)temp]=1.0;
	}

	fclose(fp);
	return labarr;
}
```

​			图像数据（每张28*28）可以看到mnist训练集的图像格式。

```c
幻数：2051      数量:60000      行:28   列:28
```



#### 			2.3  建立网络结构

```c
struct conLayer  //卷积层
{
    int L,W,H;
    double m[30][30][5];
    double b[30][30][5];
    double delta[30][30][5];

}conLayer;

struct fconLayer //全连接层
{
    int length;
    double m[1000];
    double b[1000];
    double delta[1000];
    double w[20][1000];
}fconLayer;

struct Network //总体网络结构
{
    struct conLayer Input_layer;
    struct conLayer conv_layer1;
    struct conLayer pool_layer1;
    struct conLayer filter1[5];
    struct fconLayer fcnn_input;
    struct fconLayer fcnn_w;
    struct fconLayer fcnn_outpot;
}CNN;
```

权重初始化

```c
struct conLayer init1(struct conLayer A)
{
    for(int i=0;i<30;i++)
        for(int j=0;j<30;j++)
            for(int k=0;k<5;k++)
                A.m[i][j][k]=0.01*(rand()%100);
    return A;
};

struct fconLayer init2(struct fconLayer A)
{
    for(int i=0;i<20;i++)
        for(int j=0;j<1000;j++)
            A.w[i][j]=0.01*(rand()%100);
    return A;
}
```

```c
output:

w[2][15],channel 0 = 0.730000   w[2][15],channel 1 = 0.410000   w[2][15],channel 2 = 0.230000   w[2][15],channel 3 = 0.270000   w[2][15],channel 4 = 0.950000
w[2][16],channel 0 = 0.090000   w[2][16],channel 1 = 0.860000   w[2][16],channel 2 = 0.710000   w[2][16],channel 3 = 0.230000   w[2][16],channel 4 = 0.690000
w[2][17],channel 0 = 0.640000   w[2][17],channel 1 = 0.550000   w[2][17],channel 2 = 0.340000   w[2][17],channel 3 = 0.950000   w[2][17],channel 4 = 0.530000
w[2][18],channel 0 = 0.740000   w[2][18],channel 1 = 0.200000   w[2][18],channel 2 = 0.110000   w[2][18],channel 3 = 0.790000   w[2][18],channel 4 = 0.030000
w[2][19],channel 0 = 0.100000   w[2][19],channel 1 = 0.180000   w[2][19],channel 2 = 0.510000   w[2][19],channel 3 = 0.240000   w[2][19],channel 4 = 0.760000
w[2][20],channel 0 = 0.390000   w[2][20],channel 1 = 0.710000   w[2][20],channel 2 = 0.140000   w[2][20],channel 3 = 0.620000   w[2][20],channel 4 = 0.720000
w[2][21],channel 0 = 0.520000   w[2][21],channel 1 = 0.300000   w[2][21],channel 2 = 0.210000   w[2][21],channel 3 = 0.100000   w[2][21],channel 4 = 0.850000
w[2][22],channel 0 = 0.510000   w[2][22],channel 1 = 0.710000   w[2][22],channel 2 = 0.110000   w[2][22],channel 3 = 0.830000   w[2][22],channel 4 = 0.050000
w[2][23],channel 0 = 0.870000   w[2][23],channel 1 = 0.220000   w[2][23],channel 2 = 0.550000   w[2][23],channel 3 = 0.570000   w[2][23],channel 4 = 0.730000
w[2][24],channel 0 = 0.640000   w[2][24],channel 1 = 0.610000   w[2][24],channel 2 = 0.540000   w[2][24],channel 3 = 0.020000   w[2][24],channel 4 = 0.870000
w[2][25],channel 0 = 0.300000   w[2][25],channel 1 = 0.470000   w[2][25],channel 2 = 0.220000   w[2][25],channel 3 = 0.280000   w[2][25],channel 4 = 0.510000
w[2][26],channel 0 = 0.140000   w[2][26],channel 1 = 0.750000   w[2][26],channel 2 = 0.000000   w[2][26],channel 3 = 0.070000   w[2][26],channel 4 = 0.890000
w[2][27],channel 0 = 0.620000   w[2][27],channel 1 = 0.250000   w[2][27],channel 2 = 0.070000   w[2][27],channel 3 = 0.430000   w[2][27],channel 4 = 0.790000
w[2][28],channel 0 = 0.060000   w[2][28],channel 1 = 0.250000   w[2][28],channel 2 = 0.930000   w[2][28],channel 3 = 0.540000   w[2][28],channel 4 = 0.650000
w[2][29],channel 0 = 0.200000   w[2][29],channel 1 = 0.090000   w[2][29],channel 2 = 0.190000   w[2][29],channel 3 = 0.340000   w[2][29],channel 4 = 0.500000
```



Relu激活函数

```c
double Relu(double x)
{
    return max(0.0,x);
}
```

```c
input:
5.0
output:
5.0
```



卷积(输入层与number个卷积核做卷积运算)

```c
struct conLayer conv(struct conLayer A,struct conLayer B[], int number,struct conLayer C) 
{
    memset(C.m, 0, sizeof(C.m));
    for(int i=0;i<number;i++)
    {
        B[i].L=B[i].W=5;
        B[i].H=1;
    }
    C.L=A.L-B[0].L+1;
    C.W=A.W-B[0].W+1;
    C.H=number;
    for(int num=0;num<number;num++)
        for(int i=0;i<C.L;i++)
            for(int j=0;j<C.W;j++)
            {
                for(int a=0;a<B[0].L;a++)
                    for(int b=0;b<B[0].W;b++)
                        for(int k=0;k<A.H;k++)
                        {
                            C.m[i][j][num]+=A.m[i+a][j+b][k]*B[num].m[a][b][k];
                        }
                C.m[i][j][num]=Relu(C.m[i][j][num]+C.b[i][j][num]);
            }
    return C;
}
```

```c
the cov output:
0.000000        0.000000        0.000000        0.000000        0.000000        0.191190        0.993108        1.939818        2.430516        3.165937        3.829076        3.754368        3.464425        2.855455        1.924784
        1.036288        0.761969        0.083193        0.000000        0.000000        0.000000        0.000000
        0.000000        0.000000        0.000000        0.000000        0.000000        0.000000        0.191190
        1.095034        2.679019        3.739359        4.722446        5.842698        6.531442        6.536720
        6.518714        5.485718        3.814759        2.277217        1.387583        0.388908        0.000000
        0.000000        0.000000        0.000000        0.000000        0.000000        0.000000        0.000000
        0.000000        0.058828        0.756696        2.534121        4.327615        5.198091        6.120393
        6.564112        6.755040        6.666943        7.113342        6.386698        4.434560        3.212521
        1.974296        0.696144        0.000000        0.000000        0.000000        0.000000        0.000000
        0.000000        0.000000        0.000000        0.000000        0.187113        1.651615        3.713921
        5.476397        6.032857        7.197560        7.236404        7.082192        7.616122        8.436618
        7.571404        5.435523        3.842608        2.553665        0.836993        0.000000        0.000000
        0.000000        0.000000        0.000000        0.000000        0.000000        0.000000        0.000000
        0.320231        1.741901        3.938105        5.275804        5.747190        6.118298        5.581854
        4.912824        6.011352        7.899151        7.190032        5.759354        4.143592        2.912069
        0.833982        0.000000        0.000000        0.000000        0.000000        0.000000        0.000000
        0.000000        0.000000        0.000000        0.272654        1.887043        3.770269        4.486460
```

池化层

```c
struct conLayer maxpooling(struct conLayer conv_layer,struct conLayer A)
{
    A.L=conv_layer.L/2;
    A.W=conv_layer.W/2;
    A.H=conv_layer.H;
    //printf("the maxpooling:\n");
    for(int k=0;k<conv_layer.H;k++)
        for(int i=0;i<conv_layer.L;i+=2)
            for(int j=0;j<conv_layer.W;j+=2)
                A.m[i/2][j/2][k]=(conv_layer.m[i][j][k]+conv_layer.m[i+1][j][k])+conv_layer.m[i][j+1][k]+conv_layer.m[i+1][j+1][k])/4;
    return A;
}
```

```c
input:
2	4	6	7
3	4	6	9
1	7	4	0
2	5	7	0
output:
3.25	7
3.75	2.75
```



因输出是多个，所以最后使用softmax

```c
struct fconLayer softmax(struct fconLayer A)
{
    double sum=0.0;double maxx=-100000000;
    for(int i=0;i<out;i++)
        maxx=max(maxx,A.m[i]);
    for(int i=0;i<out;i++)
        sum+=exp(A.m[i]-maxx);
    for(int i=0;i<out;i++)
    {
        A.m[i]=exp(A.m[i]-maxx)/sum;
    }
    return A;
}
```

```c
the softmax
278.031344      0.000000
273.018583      0.000000
279.442781      0.000000
250.864989      0.000000
229.903023      0.000000
390.691697      1.000000
256.588961      0.000000
271.300030      0.000000
271.711180      0.000000
275.790274      0.000000
```

如上该分类将其分至第6类



#### 2.4  训练

向前传播

```c
void forward_propagation(int num,int flag,ImgArr trainImg,double* labels)//做一次前向输出
{
    CNN.Input_layer=CNN_Input(num,CNN.Input_layer,flag,trainImg);
    CNN.conv_layer1=conv(CNN.Input_layer,CNN.filter1,5,CNN.conv_layer1);
    CNN.pool_layer1=maxpooling(CNN.conv_layer1,CNN.pool_layer1);
    CNN.fcnn_input=Classify_input(CNN.pool_layer1,CNN.fcnn_input);
    CNN.fcnn_outpot=fcnn_Mul(CNN.fcnn_input,CNN.fcnn_w,CNN.fcnn_outpot);
    CNN.fcnn_outpot=softmax(CNN.fcnn_outpot);
    for(int i=0;i<out;i++)
    {
        printf("%f\t",CNN.fcnn_outpot.m[i]);
        if(i==(int)labels[num])
            CNN.fcnn_outpot.delta[i]=CNN.fcnn_outpot.m[i]-1.0;
        else
            CNN.fcnn_outpot.delta[i]=CNN.fcnn_outpot.m[i];
    }
}
```

```c
一次传播的输出分类
0.000189        0.000001        0.000845        0.289523        0.469452        0.000000        0.239970        0.000000        0.000000        0.000019
```



向后传播

```c
void back_propagation()//反向传播算法
{
    memset(CNN.fcnn_input.delta,0,sizeof(CNN.fcnn_input.delta));
    for(int i=0;i<CNN.fcnn_input.length;i++)
    {
        for(int j=0;j<out;j++)
        {
            CNN.fcnn_input.delta[i]+=CNN.fcnn_input.m[i]*(1.0-CNN.fcnn_input.m[i])*CNN.fcnn_w.w[j][i]*CNN.fcnn_outpot.delta[j];
        }
    }
    for(int i=0;i<CNN.fcnn_input.length;i++)
    {
        for(int j=0;j<out;j++)
        {
            CNN.fcnn_w.w[j][i]-=alpha*CNN.fcnn_outpot.delta[j]*CNN.fcnn_input.m[i];
            CNN.fcnn_w.b[j]-=alpha*CNN.fcnn_outpot.delta[j];
        }
    }
    CNN.pool_layer1=pool_input(CNN.pool_layer1,CNN.fcnn_input);
    CNN.conv_layer1=pool_delta(CNN.conv_layer1,CNN.pool_layer1);//pooling误差传递
    for(int i=0;i<5;i++)
      CNN.filter1[i]=Update(CNN.filter1[i],CNN.Input_layer,CNN.conv_layer1,i);
}
```

```c
传播一次权重更新：
卷积核（以第一层为例）：
update covl:
第0层：
0.600000        0.470000        0.320000        0.540000        0.150000
0.130000        0.780000        0.440000        0.330000        0.140000
0.760000        0.840000        0.130000        0.060000        0.370000
0.510000        0.460000        0.370000        0.150000        0.520000
0.520000        0.340000        0.440000        0.970000        0.350000

update covl:
第0层：
0.599667        0.463642        0.320719        0.542076        0.149767
0.139860        0.781991        0.442236        0.333300        0.139950
0.768789        0.843945        0.131823        0.060515        0.369991
0.512010        0.460349        0.370170        0.149833        0.519891
0.520813        0.339863        0.439797        0.969808        0.349825
    
 全连接层：
w,b:
w:0.089983      b:-0.013602     w:0.08998       b:-0.01360
w:0.390000      b:-0.000080     w:0.39000       b:-0.00008
w:0.589924      b:-0.060808     w:0.58992       b:-0.06081
w:0.413908      b:-20.845637    w:0.41391       b:-20.84564
w:0.097692      b:-33.800575    w:0.09769       b:-33.80058
w:0.300122      b:71.999980     w:0.30012       b:71.99998
w:0.048373      b:-17.277844    w:0.04837       b:-17.27784
w:0.810000      b:-0.000015     w:0.81000       b:-0.00001
w:0.100000      b:-0.000022     w:0.10000       b:-0.00002
w:0.839998      b:-0.001397     w:0.84000       b:-0.00140
```



#### 2.5  测试模型

```c
for(int time=0;time<100;time++)
    {
        double err=0;
        for(int i=0;i<trainN;i++)
        {
            int i=0;
            //printf("%dth times propagation",i);
            forward_propagation(i,0,trainImg, labels);
            err-=log(CNN.fcnn_outpot.m[(int)labels[i]]);
            back_propagation();
        }
        for(int m=0;m<5;m++)
            for(int n=0;n<5;n++)
                printf("%f\t",CNN.filter1[0].m[m][n][0]);
        printf("step: %d   loss:%.5f\n",time,1.0*err/trainN);//每次记录一遍数据集的平均误差
        int sum=0;
        for(int j=0;j<testN;j++)
        {
            forward_propagation(j,1,testImg,labels1);
            int ans=-1;
            double sign=-1;
            for(int i=0;i<out;i++)
            {
                if(CNN.fcnn_outpot.m[i]>sign)
                {
                    sign=CNN.fcnn_outpot.m[i];
                    ans=i;
                }
            }
            int ans1=ans;
            int label=(int)(labels1[j]);
            if(ans1==label) sum++;
        }
	printf("\n");
	printf("sum:%d\n",sum);
    printf("step:%d   precision: %.5f\n",++step,1.0*sum/testN);
    }
```

```c
step: 3   loss:7.29560
sum:5
step:4   precision: 0.10000
...
step: 96   loss:0.01917
sum:42
step:97   precision: 0.84
```

### 3.算法实现步骤

#### 3.1  mnist.c(图像数据读取)

```C
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "mnist.h"
int ReverseInt(int i){
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
ImgArr read_Img(const char* filename) // 读入图像
{
	FILE  *fp=NULL;
	fp=fopen(filename,"rb");
	if(fp==NULL)
		printf("open file failed\n");
	assert(fp);

	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;
	//从文件中读取sizeof(magic_number) 个字符到 &magic_number
	fread((char*)&magic_number,sizeof(magic_number),1,fp);
	magic_number = ReverseInt(magic_number);
	//获取训练或测试image的个数number_of_images
	fread((char*)&number_of_images,sizeof(number_of_images),1,fp);
	number_of_images = ReverseInt(number_of_images);
	//获取训练或测试图像的高度Heigh
	fread((char*)&n_rows,sizeof(n_rows),1,fp);
	n_rows = ReverseInt(n_rows);
	//获取训练或测试图像的宽度Width
	fread((char*)&n_cols,sizeof(n_cols),1,fp);
	n_cols = ReverseInt(n_cols);
	//获取第i幅图像，保存到vec中
	printf("幻数：%d\t数量:%d\t行:%d\t列:%d\t",magic_number,number_of_images,n_rows,n_cols);
	int i,r,c;

	// 图像数组的初始化
	ImgArr imgarr=(ImgArr)malloc(sizeof(MinstImg));
	imgarr->ImgNum=number_of_images;
	imgarr->ImgPtr=(MinstImg*)malloc(number_of_images*sizeof(MinstImg));

	for(i = 0; i < number_of_images; ++i)
	{
		imgarr->ImgPtr[i].r=n_rows;
		imgarr->ImgPtr[i].c=n_cols;
		imgarr->ImgPtr[i].ImgData=(float**)malloc(n_rows*sizeof(float*));
		for(r = 0; r < n_rows; ++r)
		{
			imgarr->ImgPtr[i].ImgData[r]=(float*)malloc(n_cols*sizeof(float));
			for(c = 0; c < n_cols; ++c)
			{
				unsigned char temp = 0;
				fread((char*) &temp, sizeof(temp),1,fp);
				imgarr->ImgPtr[i].ImgData[r][c]=(float)temp/255.0;
			}
		}
	}
	fclose(fp);
	return imgarr;
}
LabelArr read_Lable(const char* filename)// 读入图像
{
	FILE  *fp=NULL;
	fp=fopen(filename,"rb");
	if(fp==NULL)
		printf("open file failed\n");
	assert(fp);
	int magic_number = 0;
	int number_of_labels = 0;
	int label_long = 10;
	//从文件中读取sizeof(magic_number) 个字符到 &magic_number
	fread((char*)&magic_number,sizeof(magic_number),1,fp);
	magic_number = ReverseInt(magic_number);
	//获取训练或测试image的个数number_of_images
	fread((char*)&number_of_labels,sizeof(number_of_labels),1,fp);
	number_of_labels = ReverseInt(number_of_labels);
	int i,l;
	// 图像标记数组的初始化
	LabelArr labarr=(LabelArr)malloc(sizeof(MinstLabel));
	labarr->LabelNum=number_of_labels;
	labarr->LabelPtr=(MinstLabel*)malloc(number_of_labels*sizeof(MinstLabel));

	for(i = 0; i < number_of_labels; ++i)
	{
		labarr->LabelPtr[i].l=10;
		labarr->LabelPtr[i].LabelData=(float*)calloc(label_long,sizeof(float));
		unsigned char temp = 0;
		fread((char*) &temp, sizeof(temp),1,fp);
		labarr->LabelPtr[i].LabelData[(int)temp]=1.0;
	}
	fclose(fp);
	return labarr;
}
char* intTochar(int i)// 将数字转换成字符串
{
	int itemp=i;
	int w=0;
	while(itemp>=10){
		itemp=itemp/10;
		w++;
	}
	char* ptr=(char*)malloc((w+2)*sizeof(char));
	ptr[w+1]='\0';
	int r; // 余数
	while(i>=10){
		r=i%10;
		i=i/10;
		ptr[w]=(char)(r+48);
		w--;
	}
	ptr[w]=(char)(i+48);
	return ptr;
}
char * combine_strings(char *a, char *b) // 将两个字符串相连
{
	char *ptr;
	int lena=strlen(a),lenb=strlen(b);
	int i,l=0;
	ptr = (char *)malloc((lena+lenb+1) * sizeof(char));
	for(i=0;i<lena;i++)
		ptr[l++]=a[i];
	for(i=0;i<lenb;i++)
		ptr[l++]=b[i];
	ptr[l]='\0';
	return(ptr);
}

void save_Img(ImgArr imgarr,char* filedir) // 将图像数据保存成文件
{
	int img_number=imgarr->ImgNum;
	int i,r;
	for(i=0;i<img_number;i++){
		const char* filename=combine_strings(filedir,combine_strings(intTochar(i),".gray"));
		FILE  *fp=NULL;
		fp=fopen(filename,"wb");
		if(fp==NULL)
			printf("write file failed\n");
		assert(fp);
		for(r=0;r<imgarr->ImgPtr[i].r;r++)
			fwrite(imgarr->ImgPtr[i].ImgData[r],sizeof(float),imgarr->ImgPtr[i].c,fp);
		fclose(fp);
	}
}
```

#### 3.2  mnist.h

```c
#ifndef __MINST_
#define __MINST_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

typedef struct MinstImg{
	int c;           // 图像宽
	int r;           // 图像高
	float** ImgData; // 图像数据二维动态数组
}MinstImg;

typedef struct MinstImgArr{
	int ImgNum;        // 存储图像的数目
	MinstImg* ImgPtr;  // 存储图像数组指针
}*ImgArr;              // 存储图像数据的数组

typedef struct MinstLabel{
	int l;            // 输出标记的长
	float* LabelData; // 输出标记数据
}MinstLabel;

typedef struct MinstLabelArr{
	int LabelNum;
	MinstLabel* LabelPtr;
}*LabelArr;              // 存储图像标记的数组

LabelArr read_Lable(const char* filename); // 读入图像标记

ImgArr read_Img(const char* filename); // 读入图像

void save_Img(ImgArr imgarr,char* filedir); // 将图像数据保存成文件

#endif
```

#### 3.3  main.c

```c
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdbool.h>
#include "mnist.h"
#define max(a,b) ( (a>b) ? a:b )
const int trainN = 200;
const int testN = 50;
const int out = 10;
const double alpha = 0.01; //learning rate
int step;
ImgArr trainImg;
ImgArr testImg;
double labels[60000];
double labels1[10000];

struct conLayer
{
    int L,W,H;
    double m[30][30][5];
    double b[30][30][5];
    double delta[30][30][5];

}conLayer;

struct fconLayer
{
    int length;
    double m[1000];
    double b[1000];
    double delta[1000];
    double w[20][1000];
}fconLayer;
struct fconLayer init2(struct fconLayer A)
{
    for(int i=0;i<20;i++)
        for(int j=0;j<1000;j++)
            A.w[i][j]=0.01*(rand()%100);
    return A;
}
struct Network
{
    struct conLayer Input_layer;
    struct conLayer conv_layer1;
    struct conLayer pool_layer1;
    struct conLayer filter1[5];
    struct fconLayer fcnn_input;
    struct fconLayer fcnn_w;
    struct fconLayer fcnn_outpot;
}CNN;
struct conLayer init1(struct conLayer A)
{
    for(int i=0;i<30;i++)
    {
        for(int j=0;j<30;j++)
        {
            for(int k=0;k<5;k++)
            {
                A.m[i][j][k]=0.01*(rand()%100);
            }
        }
    }

    return A;
};
void init()
{
    CNN.Input_layer = init1(CNN.Input_layer);
    printf("%f\n",CNN.Input_layer.m[0][0][0]);
    CNN.conv_layer1=init1(CNN.conv_layer1);
    CNN.pool_layer1=init1(CNN.pool_layer1);
    for(int i=0;i<5;i++)
        CNN.filter1[i]=init1(CNN.filter1[i]);
    CNN.fcnn_input = init2(CNN.fcnn_input);
    CNN.fcnn_w = init2(CNN.fcnn_w);
    CNN.fcnn_outpot = init2(CNN.fcnn_outpot);
}



double Relu(double x)
{
    return max(0.0,x);
}


struct fconLayer softmax(struct fconLayer A)
{
    double sum=0.0;double maxx=-100000000;
    for(int i=0;i<out;i++)
        maxx=max(maxx,A.m[i]);
    for(int i=0;i<out;i++)
        sum+=exp(A.m[i]-maxx);
    for(int i=0;i<out;i++)
    {
        A.m[i]=exp(A.m[i]-maxx)/sum;
    }
    return A;
}

double sigmod(double x)
{
    return 1.0/(1.0+exp(-x));
}


double sum(struct conLayer A,int k)
{
    double a=0;
    for(int i=0;i<A.L;i++)
        for(int j=0;j<A.W;j++)
            a+=A.delta[i][j][k];
    return a;
}

double sum1(struct conLayer A,struct conLayer B,int x,int y,int aa)
{
    double a=0;
    for(int i=0;i<A.L;i++)
        for(int j=0;j<A.W;j++)
            a+=A.delta[i][j][aa]*B.m[i+x][j+y][0];
    return a;
}

struct conLayer Update(struct conLayer A,struct conLayer B,struct conLayer C,int aa)
{
    for(int i=0;i<A.L;i++)
        for(int j=0;j<A.W;j++)
            for(int k=0;k<A.H;k++)
            {
                A.m[i][j][k]-=alpha*sum1(C,B,i,j,aa);
                C.b[i][j][k]-=alpha*sum(C,aa);
            }
    return A;
}

struct conLayer CNN_Input(int num,struct conLayer A,int flag,ImgArr trainImg)
{
    A.L=A.W=28;
    A.H=1;
    for(int i=0;i<28;i++)
    {
        for(int j=0;j<28;j++)
        {
            for(int k=0;k<1;k++)
            {
                if(flag==0)
                {
                    A.m[i][j][k]=trainImg->ImgPtr[num].ImgData[i][j];
                }
                else
                {
                    A.m[i][j][k]=trainImg->ImgPtr[num].ImgData[i][j];
                }
            }
        }
    }
    return A;
}

struct conLayer conv(struct conLayer A,struct conLayer B[], int number,struct conLayer C)
{
    memset(C.m, 0, sizeof(C.m));
    for(int i=0;i<number;i++)
    {
        B[i].L=B[i].W=5;
        B[i].H=1;
    }
    C.L=A.L-B[0].L+1;
    C.W=A.W-B[0].W+1;
    C.H=number;
    for(int num=0;num<number;num++)
        for(int i=0;i<C.L;i++)
            for(int j=0;j<C.W;j++)
            {
                for(int a=0;a<B[0].L;a++)
                    for(int b=0;b<B[0].W;b++)
                        for(int k=0;k<A.H;k++)
                        {
                            C.m[i][j][num]+=A.m[i+a][j+b][k]*B[num].m[a][b][k];
                        }
                C.m[i][j][num]=Relu(C.m[i][j][num]+C.b[i][j][num]);
            }
    return C;
}

struct fconLayer Classify_input(struct conLayer A,struct fconLayer B)
{
    int x=0;
    for(int i=0;i<A.L;i++)
        for(int j=0;j<A.W;j++)
            for(int k=0;k<A.H;k++)
                B.m[x++]=sigmod(A.m[i][j][k]);
    B.length=x;
    return B;
}

struct conLayer pool_input(struct conLayer A,struct fconLayer B)
{
    int x=1;
    for(int i=0;i<A.L;i++)
        for(int j=0;j<A.W;j++)
            for(int k=0;k<A.H;k++)
            {
                A.delta[i][j][k]=B.delta[x++];
            }
    return A;
}

struct conLayer pool_delta(struct conLayer A,struct conLayer B)
{

        double aa[A.H][A.L][A.W];
        for(int i=0;i<A.H;i++)
            for(int j=0;j<A.L;j+=2)
                for(int k=0;k<A.W;k+=2)
                {
                        aa[j][k][i] = B.delta[j/2][k/2][i]/4.0;
                        aa[j+1][k][i] = B.delta[j/2][k/2][i]/4.0;
                        aa[j][k+1][i] = B.delta[j/2][k/2][i]/4.0;
                        aa[j+1][k+1][i] = B.delta[j/2][k/2][i]/4.0;
                }
    for(int k=0;k<A.H;k++)
        for(int i=0;i<A.L;i++)
        {
            for(int j=0;j<A.W;j++)
            {
                if(A.m[i][j][k]<0)
                    A.delta[i][j][k]=0;
                else
                    A.delta[i][j][k]=A.m[i][j][k]*aa[i][j][k];
            }
        }
    return A;
}
struct conLayer maxpooling(struct conLayer conv_layer,struct conLayer A)
{
    A.L=conv_layer.L/2;
    A.W=conv_layer.W/2;
    A.H=conv_layer.H;
    for(int k=0;k<conv_layer.H;k++)
        for(int i=0;i<conv_layer.L;i+=2)
            for(int j=0;j<conv_layer.W;j+=2)
                A.m[i/2][j/2][k]=(conv_layer.m[i][j][k]+conv_layer.m[i+1][j][k]+conv_layer.m[i][j+1][k],conv_layer.m[i+1][j+1][k])/4.0;
    return A;
}

struct fconLayer fcnn_Mul(struct fconLayer A,struct fconLayer B,struct fconLayer C)
{
    memset(C.m,0,sizeof(C.m));
    C.length=out;
    for(int i=0;i<C.length;i++)
    {
        for(int j=0;j<A.length;j++)
        {
            C.m[i]+=B.w[i][j]*A.m[j];
        }
        C.m[i]+=B.b[i];
    }
    return C;
}

void forward_propagation(int num,int flag,ImgArr trainImg,double* labels)//做一次前向输出
{
    CNN.Input_layer=CNN_Input(num,CNN.Input_layer,flag,trainImg);
    CNN.conv_layer1=conv(CNN.Input_layer,CNN.filter1,5,CNN.conv_layer1);
    CNN.pool_layer1=maxpooling(CNN.conv_layer1,CNN.pool_layer1);
    CNN.fcnn_input=Classify_input(CNN.pool_layer1,CNN.fcnn_input);
    CNN.fcnn_outpot=fcnn_Mul(CNN.fcnn_input,CNN.fcnn_w,CNN.fcnn_outpot);
    CNN.fcnn_outpot=softmax(CNN.fcnn_outpot);
    for(int i=0;i<out;i++)
    {
        if(i==(int)labels[num])
            CNN.fcnn_outpot.delta[i]=CNN.fcnn_outpot.m[i]-1.0;
        else
            CNN.fcnn_outpot.delta[i]=CNN.fcnn_outpot.m[i];
    }
}
void back_propagation()//反向传播算法
{
    memset(CNN.fcnn_input.delta,0,sizeof(CNN.fcnn_input.delta));
    for(int i=0;i<CNN.fcnn_input.length;i++)
    {
        for(int j=0;j<out;j++)
        {
            CNN.fcnn_input.delta[i]+=CNN.fcnn_input.m[i]*(1.0-CNN.fcnn_input.m[i])*CNN.fcnn_w.w[j][i]*CNN.fcnn_outpot.delta[j];

        }

    }
    for(int i=0;i<CNN.fcnn_input.length;i++)
    {
 
        for(int j=0;j<out;j++)
        {

            CNN.fcnn_w.w[j][i]-=alpha*CNN.fcnn_outpot.delta[j]*CNN.fcnn_input.m[i];
            CNN.fcnn_w.b[j]-=alpha*CNN.fcnn_outpot.delta[j];

        }
    }

    CNN.pool_layer1=pool_input(CNN.pool_layer1,CNN.fcnn_input);
    CNN.conv_layer1=pool_delta(CNN.conv_layer1,CNN.pool_layer1);//pooling误差传递
    for(int i=0;i<5;i++)
        CNN.filter1[i]=Update(CNN.filter1[i],CNN.Input_layer,CNN.conv_layer1,i);
}

int main()
{
    init();
    LabelArr trainLabel=read_Lable("train-labels.idx1-ubyte");
	ImgArr trainImg=read_Img("train-images.idx3-ubyte");
	LabelArr testLabel = read_Lable("t10k-labels.idx1-ubyte");
	ImgArr testImg = read_Img("t10k-images.idx3-ubyte");
	printf("%d\n",trainLabel->LabelNum);
	double labels[trainLabel->LabelNum];
	for(int i=0;i<trainLabel->LabelNum;i++)
    {
        for(int j=0;j<10;j++){
        if(trainLabel->LabelPtr[i].LabelData[j]==1.0)
            labels[i] = (double)j;
        }
    }
    double labels1[testLabel->LabelNum];
    for(int i=0;i<testLabel->LabelNum;i++)
    {
        for(int j=0;j<10;j++){
        if(testLabel->LabelPtr[i].LabelData[j]==1.0)
            labels1[i] = (double)j;
        }
    }

    step=0;
    for(int time=0;time<100;time++)
    {
        double err=0;
        for(int i=0;i<trainN;i++)
        {
            forward_propagation(i,0,trainImg, labels);
            err-=log(CNN.fcnn_outpot.m[(int)labels[i]]);
            back_propagation();
        }
        printf("\nconvdelta:\n");
        for(int m=0;m<24;m++)
            for(int n=0;n<24;n++)
                printf("%f\t",CNN.conv_layer1.delta[m][n][0]);
        printf("\n convll:\n");

        for(int m=0;m<5;m++)
            for(int n=0;n<5;n++)
                printf("%f\t",CNN.filter1[0].m[m][n][0]);
        printf("step: %d   loss:%.5f\n",time,1.0*err/trainN);//每次记录一遍数据集的平均误差
        int sum=0;
        for(int j=0;j<testN;j++)
        {
            forward_propagation(j,1,testImg,labels1);
            int ans=-1;
            double sign=-1;
            for(int i=0;i<out;i++)
            {
                if(CNN.fcnn_outpot.m[i]>sign)
                {
                    sign=CNN.fcnn_outpot.m[i];
                    ans=i;
                }
            }
            int ans1=ans;
            int label=(int)(labels1[j]);
            if(ans1==label) sum++;
        }

	printf("\n");
	printf("sum:%d\n",sum);
    printf("step:%d   precision: %.5f\n",++step,1.0*sum/testN);
    }
    return 0;
}
```

