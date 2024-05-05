// nvcc -O3 ./matmul.cu -o matmul && ./matmul

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

void matmul_cpu(float *A, float *B, float *C, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float val = 0.0f;
      for (int k = 0; k < K; ++k) {
        val += A[k * M + m] * B[k * N + n];
      }
      C[m * N + n] = val;
    }
  }
}

#define BLOCKSIZE 64
#define TILESIZE 8
#define TILENUM (BLOCKSIZE / TILESIZE)
#define BLOCKDIM (TILENUM * TILENUM)

__global__ void matmul_kernel_1(float *A, float *B, float *C, int M, int N,
                                int K) {
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;
  const int threadRow = threadIdx.x / BLOCKSIZE;
  const int threadCol = threadIdx.x % BLOCKSIZE;
  const int m = cRow * BLOCKSIZE + threadRow;
  const int n = cCol * BLOCKSIZE + threadCol;

  if (m < M && n < N) {
    float val = 0.0f;
    for (int k = 0; k < K; ++k) {
      val += A[k * M + m] * B[k * N + n];
    }
    C[m * N + n] = val;
  }
}
void matmul_cuda_1(float *A, float *B, float *C, int M, int N, int K) {
  dim3 gridDim(ceil_div(M, BLOCKSIZE), ceil_div(N, BLOCKSIZE));
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  matmul_kernel_1<<<gridDim, blockDim>>>(A, B, C, M, N, K);
  cudaCheck(cudaGetLastError());
}

__global__ void matmul_kernel_2(float *A, float *B, float *C, int M, int N,
                                int K) {
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;
  const int threadRow = threadIdx.x / BLOCKSIZE;
  const int threadCol = threadIdx.x % BLOCKSIZE;

  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  A += cRow * BLOCKSIZE;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  float val = 0.0f;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * M + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
    __syncthreads();

    A += BLOCKSIZE * M;
    B += BLOCKSIZE * N;

    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      val += As[dotIdx * BLOCKSIZE + threadRow] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    __syncthreads();
  }
  C[threadRow * N + threadCol] = val;
}
void matmul_cuda_2(float *A, float *B, float *C, int M, int N, int K) {
  dim3 gridDim(ceil_div(M, BLOCKSIZE), ceil_div(N, BLOCKSIZE));
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  matmul_kernel_2<<<gridDim, blockDim>>>(A, B, C, M, N, K);
  cudaCheck(cudaGetLastError());
}

__global__ void matmul_kernel_3(float *A, float *B, float *C, int M, int N,
                                int K) {
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;
  const int threadRow = threadIdx.x / (BLOCKSIZE / TILESIZE);
  const int threadCol = threadIdx.x % (BLOCKSIZE / TILESIZE);
  const int copySize = BLOCKDIM / TILESIZE;
  const int copyUnit = BLOCKSIZE / copySize;
  const int innerRow = threadIdx.x / copySize;
  const int innerCol = threadIdx.x % copySize;

  __shared__ float As[TILESIZE * BLOCKSIZE];
  __shared__ float Bs[TILESIZE * BLOCKSIZE];

  float threadResults[TILESIZE * TILESIZE] = {0.0f};
  float regM[TILESIZE] = {0.0f};
  float regN[TILESIZE] = {0.0f};

  A += cRow * BLOCKSIZE;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  for (int bkIdx = 0; bkIdx < K; bkIdx += TILESIZE) {
    for (int loadOffset = 0; loadOffset < copyUnit; ++loadOffset) {
      As[innerRow * BLOCKSIZE + innerCol * copyUnit + loadOffset] =
          A[innerRow * M + innerCol * copyUnit + loadOffset];
    }
    for (int loadOffset = 0; loadOffset < copyUnit; ++loadOffset) {
      Bs[innerRow * BLOCKSIZE + innerCol * copyUnit + loadOffset] =
          B[innerRow * N + innerCol * copyUnit + loadOffset];
    }

    __syncthreads();

    A += TILESIZE * M;
    B += TILESIZE * N;

    for (int dotIdx = 0; dotIdx < TILESIZE; ++dotIdx) {
      for (int i = 0; i < TILESIZE; ++i) {
        regM[i] = As[dotIdx * BLOCKSIZE + threadRow * TILESIZE + i];
      }
      for (int i = 0; i < TILESIZE; ++i) {
        regN[i] = Bs[dotIdx * BLOCKSIZE + threadCol * TILESIZE + i];
      }
      for (int resIdxM = 0; resIdxM < TILESIZE; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TILESIZE; ++resIdxN) {
          threadResults[resIdxM * TILESIZE + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  for (int resIdxM = 0; resIdxM < TILESIZE; ++resIdxM) {
    for (int resIdxN = 0; resIdxN < TILESIZE; ++resIdxN) {
      C[(threadRow * TILESIZE + resIdxM) * N + threadCol * TILESIZE + resIdxN] =
          threadResults[resIdxM * TILESIZE + resIdxN];
    }
  }
}
void matmul_cuda_3(float *A, float *B, float *C, int M, int N, int K) {
  dim3 gridDim(ceil_div(M, BLOCKSIZE), ceil_div(N, BLOCKSIZE));
  dim3 blockDim(BLOCKDIM);
  matmul_kernel_3<<<gridDim, blockDim>>>(A, B, C, M, N, K);
  cudaCheck(cudaGetLastError());
}

__global__ void matmul_kernel_4(float *A, float *B, float *C, int M, int N,
                                int K) {
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;
  const int threadRow = threadIdx.x / (BLOCKSIZE / TILESIZE);
  const int threadCol = threadIdx.x % (BLOCKSIZE / TILESIZE);
  const int copySize = BLOCKDIM / TILESIZE;
  const int copyUnit = BLOCKSIZE / copySize;
  const int innerRow = threadIdx.x / copySize;
  const int innerCol = threadIdx.x % copySize;

  __shared__ float As[TILESIZE * BLOCKSIZE];
  __shared__ float Bs[TILESIZE * BLOCKSIZE];

  float threadResults[TILESIZE * TILESIZE] = {0.0f};
  float regM[TILESIZE] = {0.0f};
  float regN[TILESIZE] = {0.0f};

  A += cRow * BLOCKSIZE;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  for (int bkIdx = 0; bkIdx < K; bkIdx += TILESIZE) {
    for (int loadOffset = 0; loadOffset < copyUnit; loadOffset += 4) {
      reinterpret_cast<float4 *>(
          &As[innerRow * BLOCKSIZE + innerCol * copyUnit + loadOffset])[0] =
          reinterpret_cast<float4 *>(
              &A[innerRow * M + innerCol * copyUnit + loadOffset])[0];
    }
    for (int loadOffset = 0; loadOffset < copyUnit; loadOffset += 4) {
      reinterpret_cast<float4 *>(
          &Bs[innerRow * BLOCKSIZE + innerCol * copyUnit + loadOffset])[0] =
          reinterpret_cast<float4 *>(
              &B[innerRow * N + innerCol * copyUnit + loadOffset])[0];
    }
    __syncthreads();

    A += TILESIZE * M;
    B += TILESIZE * N;

    for (int dotIdx = 0; dotIdx < TILESIZE; ++dotIdx) {
      for (int i = 0; i < TILESIZE; ++i) {
        regM[i] = As[dotIdx * BLOCKSIZE + threadRow * TILESIZE + i];
      }
      for (int i = 0; i < TILESIZE; ++i) {
        regN[i] = Bs[dotIdx * BLOCKSIZE + threadCol * TILESIZE + i];
      }
      for (int resIdxM = 0; resIdxM < TILESIZE; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TILESIZE; ++resIdxN) {
          threadResults[resIdxM * TILESIZE + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  for (int resIdxM = 0; resIdxM < TILESIZE; ++resIdxM) {
    for (int resIdxN = 0; resIdxN < TILESIZE; resIdxN += 4) {
      reinterpret_cast<float4 *>(&C[(threadRow * TILESIZE + resIdxM) * N +
                                    threadCol * TILESIZE + resIdxN])[0] =
          reinterpret_cast<float4 *>(
              &threadResults[resIdxM * TILESIZE + resIdxN])[0];
    }
  }
}
void matmul_cuda_4(float *A, float *B, float *C, int M, int N, int K) {
  dim3 gridDim(ceil_div(M, BLOCKSIZE), ceil_div(N, BLOCKSIZE));
  dim3 blockDim(BLOCKDIM);
  matmul_kernel_4<<<gridDim, blockDim>>>(A, B, C, M, N, K);
  cudaCheck(cudaGetLastError());
}

int main(int argc, char **argv) {
  srand(0);

  int M = 512;
  int N = 1024;
  int K = 768;

  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceIdx);
  printf("Device %d: %s\n", deviceIdx, deviceProp.name);

  float *A = make_random_float(K * M);
  float *B = make_random_float(K * N);
  float *C = (float *)malloc(M * N * sizeof(float));

  float *A_d;
  float *B_d;
  float *C_d;

  cudaCheck(cudaMalloc(&A_d, K * M * sizeof(float)));
  cudaCheck(cudaMalloc(&B_d, K * N * sizeof(float)));
  cudaCheck(cudaMalloc(&C_d, M * N * sizeof(float)));
  cudaCheck(cudaMemcpy(A_d, A, K * M * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  matmul_cpu(A, B, C, M, N, K);
  matmul_cuda_4(A_d, B_d, C_d, M, N, K);
  validate_result(C_d, C, "out", M * N, 1e-1f);

  int repeat_times = 100;
  float elapsed_time =
      benchmark_kernel(repeat_times, matmul_cuda_4, A_d, B_d, C_d, M, N, K);

  float tflops = (float)M * N * K * 2 / elapsed_time * 1e3f / 1e12f;
  float max_tflops = 20.31f;
  printf("time %.4f ms | tflops %.2f (%.2f %%)\n", elapsed_time, tflops,
         (tflops / max_tflops * 100.0f));

  free(A);
  free(B);
  free(C);
  cudaCheck(cudaFree(A_d));
  cudaCheck(cudaFree(B_d));
  cudaCheck(cudaFree(C_d));

  return 0;
}