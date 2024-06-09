// nvcc -o test ./reducemean.cu && ./test

#include <cuda_runtime.h>

#include <iostream>

// Kernel function to perform reduction (mean) on a 2D matrix along axis=1
// (row-wise)
__global__ void reduceMeanAxis1(float *input, float *output, int rows,
                                int cols) {
  extern __shared__ float sharedData[];

  int row = blockIdx.x;
  int tid = threadIdx.x;

  // Each thread loads one element from the row into shared memory
  sharedData[tid] = (tid < cols) ? input[row * cols + tid] : 0.0f;
  __syncthreads();

  // Perform reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sharedData[tid] += sharedData[tid + s];
    }
    __syncthreads();
  }

  // Write the result of this row to the output array
  if (tid == 0) {
    output[row] = sharedData[0] / cols;
  }
}

// Host function to call the kernel and compute the mean of the rows
void meanMatrixAxis1(float *input, float *output, int rows, int cols) {
  float *d_input, *d_output;
  size_t inputSize = rows * cols * sizeof(float);
  size_t outputSize = rows * sizeof(float);
  size_t sharedMemSize = cols * sizeof(float);  // Each block sums one row

  // Allocate device memory
  cudaMalloc((void **)&d_input, inputSize);
  cudaMalloc((void **)&d_output, outputSize);

  // Copy data from host to device
  cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);

  // Launch the reduction kernel
  reduceMeanAxis1<<<rows, cols, sharedMemSize>>>(d_input, d_output, rows, cols);

  // Copy the results back to the host
  cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);
}

int main() {
  int rows = 4;
  int cols = 4;
  int totalElements = rows * cols;
  float *h_input = (float *)malloc(totalElements * sizeof(float));
  float *h_output = (float *)malloc(rows * sizeof(float));

  // Initialize 2D matrix
  for (int i = 0; i < totalElements; ++i) {
    h_input[i] = static_cast<float>(i + 1);  // Fill with 1, 2, ..., n
  }

  // Calculate mean using CUDA
  meanMatrixAxis1(h_input, h_output, rows, cols);

  // Output the result
  std::cout << "Row-wise mean: ";
  for (int i = 0; i < rows; ++i) {
    std::cout << h_output[i] << " ";
  }
  std::cout << std::endl;

  // Free host memory
  free(h_input);
  free(h_output);

  return 0;
}