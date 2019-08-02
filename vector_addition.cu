#include <cuda.h>
#include <stdio.h>
// #include <cuda_rentime.h>

#define N 1024

__global__ void VecAdd(float *A, float *B, float *C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

int main() {
  float h_A[N], h_B[N], h_C[N];
  size_t size = N * sizeof(float);
  float *d_A;
  cudaMalloc((void **)&d_A, size);
  float *d_B;
  cudaMalloc((void **)&d_B, size);
  float *d_C;
  cudaMalloc((void **)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = N;
  int threadsPerGrid = 1;

  VecAdd<<<threadsPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
