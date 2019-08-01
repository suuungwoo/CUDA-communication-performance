/////////////////////////////////////////////////////////////////////////  
//
//  備考:このコードでは二次元行列同士の内積を行い実行時間を観測する。
//
/////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

////////////////////////
//  ↓行列構造体定義
////////////////////////
typedef struct matrix{
    int col;
    int row;
    double* element;
}Matrix;

//////////////////////////////////
//  ↓内積を行うカーネル関数
//////////////////////////////////
__global__
void gpu_dot(const double* x,const double* w,double* out,const int* x_row,const int* w_row, const int* w_col){

    int i,j,k;
    double sum = 0;

    //変数iには現在実行中のスレッドの固有の番号(y軸)が入る(スレッド数は結果保存用行列構造体の行数個ある)
    //変数jには現在実行中のスレッドの固有の番号(x軸)が入る(スレッド数は結果保存用行列構造体の列数個ある)
    i = blockIdx.y * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + threadIdx.x;

    for(k=0; k<(*w_col); k++){
        sum += x[i*(*x_row)+k] * w[k*(*w_row)+j];
    }
    out[i*(*w_row)+j] = sum;
}

////////////////////////////////////////////////////
//  ↓行列構造体を操作する関数のプロトタイプ宣言
////////////////////////////////////////////////////
//↓コンストラクタ
void Matrix_constructor(Matrix* self,const int col,const int row);

//↓行列の要素に要素番号代入
void Matrix_init(Matrix* self);

//↓行列をゼロクリア
void Matrix_zeros(Matrix* self);

//↓行列の中身をすべて表示
void Matrix_print(Matrix* self);

//行列構造体内の要素開放
void Matrix_free(Matrix* self);

///////////////////////
//  ↓メイン関数
///////////////////////

int main(){

    //↓タイマー用変数宣言
    time_t start,stop;

    //↓行列構造体宣言
    Matrix x;
    Matrix w;
    Matrix out;//←計算結果保存用

    //↓行列構造体のコンストラクタ
    Matrix_constructor(&x,2000,3000);
    Matrix_constructor(&w,3000,5000);
    Matrix_constructor(&out,2000,5000);

    //↓入力用の行列に数値代入
    Matrix_init(&x);
    Matrix_init(&w);

    //↓出力保存用行列を0クリア
    Matrix_zeros(&out);

    //↓カーネル関数の引数に使う変数宣言(配列として使う)
    double* gpu_x;
    double* gpu_w;
    double* gpu_out;

    //↓カーネル関数の引数に使う変数宣言(定数として使う)
    int* x_row;
    int* w_row;
    int* w_col;

    //↓cudaMallocでかかる時間測定開始
    start = clock();

    //↓カーネルの引数に使う配列の動的確保
    cudaMalloc(&gpu_x,sizeof(double)*x.col*x.row);
    cudaMalloc(&gpu_w,sizeof(double)*w.col*w.row);
    cudaMalloc(&gpu_out,sizeof(double)*out.col*out.row);

    //↓カーネルの引数に使う定数の動的確保
    cudaMalloc(&x_row,sizeof(int));
    cudaMalloc(&w_row,sizeof(int));
    cudaMalloc(&w_col,sizeof(int));

    //↓cudaMallocでかかる時間測定終了
    stop = clock();

    //↓cudaMallocでかかる時間表示
    printf("cudaMalloc:%lf秒\n",(double)(stop-start)/CLOCKS_PER_SEC);

    //↓cudaMemcpyでかかる時間測定開始
    start = clock();


    //↓計算で使う行列の中身をカーネルの引数で使う変数へコピー
    cudaMemcpy(gpu_x,x.element,sizeof(double)*x.col*x.row,cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_w,w.element,sizeof(double)*w.col*w.row,cudaMemcpyHostToDevice);

    //↓計算で使う定数の中身をカーネルの引数で使う変数へコピー
    cudaMemcpy(x_row,&(x.row),sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(w_row,&(w.row),sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(w_col,&(w.col),sizeof(int),cudaMemcpyHostToDevice);

    //↓cudaMemcpyでかかる時間測定終了
    stop = clock();

    //↓cudaMemcpyでかかる時間表示
    printf("cudaMemcpy(Host_to_Device):%lf秒\n",(double)(stop-start)/CLOCKS_PER_SEC);

    //↓内積計算でかかる時間測定開始
    start = clock();

    //↓内積計算実行
    gpu_dot<<<dim3(out.row,out.col,1),dim3(1,1,1)>>>(gpu_x,gpu_w,gpu_out,x_row,w_row,w_col);

    cudaDeviceSynchronize();


    //↓内積計算でかかる時間測定終了
    stop = clock();

    //↓内積計算でかかる時間表示
    printf("内積計算:%lf秒\n",(double)(stop-start)/CLOCKS_PER_SEC);

    //↓カーネル用変数からホスト用変数に内容をコピーするのにかかる時間測定開始
    start = clock();

    //↓カーネル用変数からホスト用変数に内容をコピー
    cudaMemcpy(out.element,gpu_out,sizeof(double)*out.col*out.row,cudaMemcpyDeviceToHost);

    //↓カーネル用変数からホスト用変数に内容をコピーするのにかかる時間測定終了
    stop = clock();

    //↓カーネル用変数からホスト用変数に内容をコピーするのにかかる時間表示
    printf("cudaMemcpy(Devise_to_Host):%lf秒\n",(double)(stop-start)/CLOCKS_PER_SEC);

    //↓ホスト側デバイス側共に動的確保した領域開放
    cudaFree(gpu_x);
    cudaFree(gpu_w);
    cudaFree(gpu_out);

    cudaFree(x_row);
    cudaFree(w_row);
    cudaFree(w_col);

    Matrix_free(&x);
    Matrix_free(&w);
    Matrix_free(&out);

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//  ↓ここから行列構造体を操作する関数の実装(関数の解説は上記のプロトタイプ宣言部分に記載)
///////////////////////////////////////////////////////////////////////////////////////////////

void Matrix_constructor(Matrix* self,const int col,const int row){
    self->col = col;
    self->row = row;

    self->element = (double*)malloc(sizeof(double)*col*row);
}

void Matrix_init(Matrix* self){
    for(int i=0;i<self->col;i++){
        for(int j=0;j<self->row;j++){
            self->element[i*self->row+j] = i*self->row+j;
        }
    }
}

void Matrix_zeros(Matrix* self){
    for(int i=0;i<self->col;i++){
        for(int j=0;j<self->row;j++){
            self->element[i*self->row+j] = 0;
        }
    }
}

void Matrix_print(Matrix* self){
    for(int i=0;i<self->col;i++){
        for(int j=0;j<self->row;j++){
            printf("[%lf]",self->element[i*self->row+j]);
        }
        printf("\n");
    }
}

void Matrix_free(Matrix* self){
    free(self->element);
    self->element = NULL;
}
