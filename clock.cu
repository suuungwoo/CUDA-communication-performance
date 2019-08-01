#include <stdio.h>
#include <cuda.h>

__global__ void my_kernel(long long *clocks)
{
    // 開始時間を記録
    long long start = clock64();
    printf("Start Clock : %ld\n", start);
    // 終了時間を記録
    clocks[0] = clock64() - start;
}

int main()
{
    int clock_rate = 0;
    int device = 0;

    long long *clock_data;
    long long *host_data;

    long long size;
    int i;

    // クロックレートを取る
    cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device);

    // 初期化
    host_data = (long long *)malloc(sizeof(long long));
    cudaMalloc(&clock_data, sizeof(long long));

    my_kernel<<<1, 1>>>(clock_data);

    for (i=0; i<=16;i++){

      // GPUで測定したクロックの差を持ってくる
      cudaMemcpy(host_data, clock_data, i, cudaMemcpyDeviceToHost);
      
      // ms単位で時間に換算する。
      printf("Elapsed clock cycles: %lld, clock rate: %d kHz\n", host_data[0], clock_rate);
      printf("Execution time: %f ms\nメッセージ長：%d\n", host_data[0]/(float)clock_rate, i);
    }
    return 0;
}
