#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <math.h>

int row,col,n_codebooks;

extern int get_row(char *filename);
extern int get_col(char *filename);
extern void get_two_dimension(char *line, double **dataset, char *filename);
extern float evaluate_algorithm(double **dataset, int n_folds, int fold_size, float l_rate, int n_epoch);


double euclidean_distance(double*row1, double*row2){
    int i;
    double distance = 0.0;
    for (i=0;i<col-1;i++){
        distance =distance+ (row1[i] - row2[i])*(row1[i] - row2[i]);
    }
    return sqrt(distance);
}

//Locate the best matching unit
int get_best_matching_unit(double**codebooks, double*test_row,int n_codebooks){
    double dist_min,dist;
    int i,min=0;
    dist_min = euclidean_distance(codebooks[0], test_row);
    for (i=0;i< n_codebooks;i++){
        dist = euclidean_distance(codebooks[i], test_row);
        if(dist < dist_min){
        dist_min=dist;
        min=i;
        }
    }
    //bmu=codebooks[min];
    return min;
}

// Make a prediction with codebook vectors
float predict(double**codebooks, double*test_row){
    int min;
    min=get_best_matching_unit(codebooks,test_row,n_codebooks);
    return (float)codebooks[min][col-1];
}

// Create random codebook vectors
double** random_codebook(double**train,int n_codebooks,int fold_size){
    int i,j,r;
    int n_folds=(int)(row/fold_size);
    double **codebooks=(double **)malloc(n_codebooks * sizeof(int*));
    for ( i=0;i < n_codebooks; ++i){
		codebooks[i] = (double *)malloc(col * sizeof(double));
	};
    srand((unsigned)time(NULL));
    for(i=0;i<n_codebooks;i++){

        for(j=0;j<col;j++){
            //srand((unsigned int)time(0));
            r=rand()%((n_folds-1)*fold_size);
            //printf(" r%d",r);
            codebooks[i][j]=train[r][j];
        }
    }
    return codebooks;
}

double** train_codebooks(double**train, double l_rate, int n_epoch,int n_codebooks,int fold_size){
    int i,j,k,min=0;
    double error,rate=0.0;
    int n_folds=(int)(row/fold_size);
    double **codebooks=(double **)malloc(n_codebooks * sizeof(int*));
    for ( i=0;i < n_codebooks; ++i){
		codebooks[i] = (double *)malloc(col * sizeof(double));
	};
    codebooks=random_codebook(train,n_codebooks,fold_size);

    for (i=0;i<n_epoch;i++){
        rate = l_rate * (1.0-(i/(double)n_epoch));
        //printf("%d ",i);
        for(j=0;j<fold_size*(n_folds-1);j++){
            //printf("%d ",i);
            min=get_best_matching_unit(codebooks, train[j],n_codebooks);

            for (k=0;k<col-1;k++){
                error = train[j][k] - codebooks[min][k];
                if (fabs(codebooks[min][col-1] - train[j][col-1])<1e-13){
                    codebooks[min][k] =codebooks[min][k]+ rate * error;}
                else{codebooks[min][k] = codebooks[min][k]-rate * error;}

                }
            }
        }

    return codebooks;
}

float* get_test_prediction(double **train, double **test, float l_rate, int n_epoch, int fold_size)
{
    int i;
    double **codebooks=(double **)malloc(n_codebooks * sizeof(int*));
    for ( i=0;i < n_codebooks; ++i){
		codebooks[i] = (double *)malloc(col * sizeof(double));
	};
    float *predictions=(float*)malloc(fold_size*sizeof(float));//预测集的行数就是数组prediction的长度

    codebooks=train_codebooks(train,l_rate,n_epoch,n_codebooks,fold_size);
    for(i=0;i<fold_size;i++)
    {
        predictions[i]=predict(codebooks,test[i]);
    }
	return predictions;
}


int main()
{
    char filename[] = "ionosphere-full.csv";
    char line[1024];
    row = get_row(filename);
    col = get_col(filename);

    int i;
	double **dataset = (double **)malloc(row * sizeof(int *));
    for ( i=0;i < row; ++i){
		dataset[i] = (double *)malloc(col * sizeof(double));
	}

    get_two_dimension(line,dataset,filename);

    int n_folds=5;
    double l_rate=0.3;
    int n_epoch=50;
    int fold_size=(int)(row/n_folds);
    n_codebooks = 20;

    evaluate_algorithm(dataset, n_folds, fold_size, l_rate, n_epoch);

    return 0;

}
