#include "process.cuh"
#include "cuda_runtime.h"
#include <cuda.h>

//カーネル　指定したスレッド数分だけ動く！
__global__ void kernel( int* pSrc1, int *pSrc2, int *pResult, int length )
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; //自分のスレッドのindex

	if (idx >= length) {
		return; //配列をはみ出る場合は無視
	}

	pResult[idx] = pSrc1[idx] + pSrc2[idx]; //加算

	return;
}

void Process()
{
	int length = 1024; 
	size_t size = sizeof(int) * length; 
	

	//ホストメモリのポインタ
	int* pHostSrc1;		//数値１
	int* pHostSrc2;		//数値２
	int* pHostResult;	//加算結果
	//デバイスメモリのポインタ
	int* pDevSrc1;		//数値１
	int* pDevSrc2;		//数値２
	int* pDevResult;	//加算結果


	//ホストメモリの確保
	cudaMallocHost(&pHostSrc1, size);	
	cudaMallocHost(&pHostSrc2, size);	
	cudaMallocHost(&pHostResult, size);	
	//デバイスメモリの確保
	cudaMalloc(&pDevSrc1, size);		
	cudaMalloc(&pDevSrc2, size);		
	cudaMalloc(&pDevResult, size);		

	//ホストメモリに適当な値を設定
	for (int n = 0; n < length; n++) {
		pHostSrc1[n] = n;
		pHostSrc2[n] = n;
	}

	//ホスト->デバイスへ数値を転送
	cudaMemcpy(pDevSrc1, pHostSrc1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(pDevSrc2, pHostSrc2, size, cudaMemcpyHostToDevice);

	//カーネル実行
	dim3 block(128, 1, 1);
	dim3 grid((length + 128 - 1 ) / 128, 1, 1);　//length数だけスレッドが生成されるようにしている
	kernel <<<grid, block>>> (pDevSrc1, pDevSrc2, pDevResult, length);

	//デバイス->ホストへ結果を転送
	cudaMemcpy(pHostResult, pDevResult, size, cudaMemcpyDeviceToHost);

	//デバイスメモリの開放
	cudaFree(pDevSrc1);
	cudaFree(pDevSrc2);
	cudaFree(pDevResult);
	//ホストメモリの開放
	cudaFreeHost(pHostSrc1);
	cudaFreeHost(pHostSrc2);
	cudaFreeHost(pHostResult);

	cudaDeviceReset();
}
