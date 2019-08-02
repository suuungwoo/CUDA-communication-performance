#include <stdio.h>
#include <stdlib.h>

__global__ void VecAdd(float *A, float *B, float *C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

int main() {
  long int N;
  FILE *outputfile;
  outputfile = fopen("sync.csv", "w"); // ファイルを書き込み用にオープン(開く)
  if (outputfile == NULL) {  // オープンに失敗した場合
    printf("cannot open\n"); // エラーメッセージを出して
    exit(1);                 // 異常終了
  }

  for (N = 2; N <= 1024000; N *= 2) {
    float h_A[N], h_B[N], h_C[N];
    size_t size = N * sizeof(float);
    int i;

    time_t start_download, stop_download, start_kernel, stop_kernel;

    float *d_A;
    cudaMalloc((void **)&d_A, size);
    float *d_B;
    cudaMalloc((void **)&d_B, size);
    float *d_C;
    cudaMalloc((void **)&d_C, size);

    srand((unsigned)time(NULL));
    for (i = 0; i < N; i++) {
      h_A[i] = rand() % 10;
      h_B[i] = rand() % 10;
    }

    start_download = clock();
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    stop_download = clock();

    int threadsPerBlock = N;
    int threadsPerGrid = 1;

    start_kernel = clock();
    VecAdd<<<threadsPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    stop_kernel = clock();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    fprintf(outputfile, "%ld,%f,%f\n", N * 8,
            (double)(stop_download - start_download),
            (double)(stop_kernel - start_kernel));
  }
  fclose(outputfile);
  return 0;
}
