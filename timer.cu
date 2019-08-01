#include <stdio.h>
#include <cuda.h>

__global__ void my_kernel(long long int *time)
{
    long long int start, stop;
    // 開始時間を記録
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(start));

    printf("Some Event... \n");

    // 終了時間を記録
    asm volatile("mov.u64  %0, %globaltimer;" : "=l"(stop));

    time[0] = stop - start;
}

int main()
{
    long long int *time_data;
    long long int *host_data;
    // int i;
    int size = 100000000;

    // for (i=1;i<=1000; i*=10){
    // 初期化
    host_data = (long long *)malloc(sizeof(long long int)*size);
    cudaMalloc(&time_data, sizeof(long long int)*size);
    printf("%lu\n", sizeof(long long int)*size);
    
    my_kernel<<<1, 1>>>(time_data);


    cudaMemcpy(host_data, time_data, sizeof(long long int)*size, cudaMemcpyDeviceToHost);

    // ns単位でのstartとstopの差
    printf("Execution time: %lld ns\n", host_data[0]);
    // }
    return 0;
}
