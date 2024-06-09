#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

// Kernel function to compute mean and variance
__global__ void computeMeanVariance(float *input, float *mean, float *variance,
                                    int rows, int cols) {
  extern __shared__ float sharedData[];

  int tid = threadIdx.x;
  int row = blockIdx.x;
  float *sharedMean = sharedData;
  float *sharedVar = sharedData + blockDim.x;

  // Compute mean
  sharedMean[tid] = (tid < cols) ? input[row * cols + tid] : 0.0f;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sharedMean[tid] += sharedMean[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    mean[row] = sharedMean[0] / cols;
  }
  __syncthreads();

  // Compute variance
  float meanVal = mean[row];
  sharedVar[tid] = (tid < cols) ? (input[row * cols + tid] - meanVal) *
                                      (input[row * cols + tid] - meanVal)
                                : 0.0f;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sharedVar[tid] += sharedVar[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    variance[row] = sharedVar[0] / cols;
  }
}

// Kernel function to normalize the input and apply scale and shift
__global__ void layerNorm(float *input, float *output, float *mean,
                          float *variance, float *gamma, float *beta, int rows,
                          int cols) {
  int tid = threadIdx.x;
  int row = blockIdx.x;

  float meanVal = mean[row];
  float varianceVal = variance[row];
  float stdDev =
      sqrt(varianceVal + 1e-5f);  // Adding epsilon for numerical stability

  if (tid < cols) {
    int index = row * cols + tid;
    output[index] = gamma[tid] * (input[index] - meanVal) / stdDev + beta[tid];
  }
}

// Host function to call the kernel
void layerNormForward(float *input, float *output, float *gamma, float *beta,
                      int rows, int cols) {
  float *d_input, *d_output, *d_mean, *d_variance, *d_gamma, *d_beta;
  size_t inputSize = rows * cols * sizeof(float);
  size_t outputSize = rows * cols * sizeof(float);
  size_t meanVarianceSize = rows * sizeof(float);
  size_t gammaBetaSize = cols * sizeof(float);

  // Allocate device memory
  cudaMalloc((void **)&d_input, inputSize);
  cudaMalloc((void **)&d_output, outputSize);
  cudaMalloc((void **)&d_mean, meanVarianceSize);
  cudaMalloc((void **)&d_variance, meanVarianceSize);
  cudaMalloc((void **)&d_gamma, gammaBetaSize);
  cudaMalloc((void **)&d_beta, gammaBetaSize);

  // Copy data from host to device
  cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, gamma, gammaBetaSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, beta, gammaBetaSize, cudaMemcpyHostToDevice);

  // Compute mean and variance
  computeMeanVariance<<<rows, cols, 2 * cols * sizeof(float)>>>(
      d_input, d_mean, d_variance, rows, cols);

  // Normalize and apply scale and shift
  layerNorm<<<rows, cols>>>(d_input, d_output, d_mean, d_variance, d_gamma,
                            d_beta, rows, cols);

  // Copy the results back to the host
  cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_mean);
  cudaFree(d_variance);
  cudaFree(d_gamma);
  cudaFree(d_beta);
}

int main() {
  int rows = 2;
  int cols = 4;
  int totalElements = rows * cols;
  float *h_input = (float *)malloc(totalElements * sizeof(float));
  float *h_output = (float *)malloc(totalElements * sizeof(float));
  float *h_gamma = (float *)malloc(cols * sizeof(float));
  float *h_beta = (float *)malloc(cols * sizeof(float));

  // Initialize input matrix
  for (int i = 0; i < totalElements; ++i) {
    h_input[i] = static_cast<float>(i + 1);  // Fill with 1, 2, ..., n
  }

  // Initialize gamma and beta
  for (int i = 0; i < cols; ++i) {
    h_gamma[i] = 1.0f;  // Scale
    h_beta[i] = 0.0f;   // Shift
  }

  // Perform layer normalization using CUDA
  layerNormForward(h_input, h_output, h_gamma, h_beta, rows, cols);

  // Output the result
  std::cout << "Layer Normalized Output: ";
  for (int i = 0; i < totalElements; ++i) {
    std::cout << h_output[i] << " ";
  }
  std::cout << std::endl;

  // Free host memory
  free(h_input);
  free(h_output);
  free(h_gamma);
  free(h_beta);

  return 0;
}