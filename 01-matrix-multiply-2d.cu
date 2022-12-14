#include <stdio.h>
#include <time.h>

#define N  512

__global__ void matrixMulGPU( int * a, int * b, int * c )
{
  int val;
  int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
  int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
  
  int stride_x = blockDim.x * gridDim.x;
  
  for( int row = idx_x; row < N; row+= stride_x ) {
    for( int col = idx_y; col < N; col += stride_x ) {
      val = 0;
      for ( int k = 0; k < N; ++k )
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
    }
  }
}

/*
 * This CPU function already works, and will run to create a solution matrix
 * against which to verify your work building out the matrixMulGPU kernel.
 */

void matrixMulCPU( int * a, int * b, int * c )
{
  int val = 0;

  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      val = 0;
      for ( int k = 0; k < N; ++k )
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
    }
}

int main()
{
  int *a, *b, *c_cpu, *c_gpu; // Allocate a solution matrix for both the CPU and the GPU operations

  int size = N * N * sizeof (int); // Number of bytes of an N x N matrix
  
  double cpu_time = 0.0;
  double gpu_time = 0.0;

  // Allocate memory
  cudaMallocManaged (&a, size);
  cudaMallocManaged (&b, size);
  cudaMallocManaged (&c_cpu, size);
  cudaMallocManaged (&c_gpu, size);

  // Initialize memory; create 2D matrices
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      a[row*N + col] = row;
      b[row*N + col] = col+2;
      c_cpu[row*N + col] = 0;
      c_gpu[row*N + col] = 0;
    }

  /*
   * Assign `threads_per_block` and `number_of_blocks` 2D values
   * that can be used in matrixMulGPU above.
   */
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
  
  cudaMemPrefetchAsync(a, size, deviceId);
  cudaMemPrefetchAsync(b, size, deviceId);
  cudaMemPrefetchAsync(c_gpu, size, deviceId);  

  dim3 threads_per_block(8, 8, 1);
  dim3 number_of_blocks(numberOfSMs, numberOfSMs, 1);
  
 // dim3 threads_per_block(1, 1, 1);
 // dim3 number_of_blocks(1, 1, 1);
  
  int nIter = 20;
  clock_t begin = clock();
  
  for (int j = 0; j < nIter; j++) {
  matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu );
  }
  cudaDeviceSynchronize();
  
  clock_t mid = clock();
  
  cudaMemPrefetchAsync(c_gpu, size, cudaCpuDeviceId);

  // Call the CPU version to check our work
  matrixMulCPU( a, b, c_cpu );
  
  clock_t end = clock();

  // Compare the two answers to make sure they are equal
  bool error = false;
  for( int row = 0; row < N && !error; ++row )
    for( int col = 0; col < N && !error; ++col )
      if (c_cpu[row * N + col] != c_gpu[row * N + col])
      {
        printf("FOUND ERROR at c[%d][%d]\n", row, col);
        error = true;
        break;
      }
  if (!error)
    printf("Success!\n");
    
  gpu_time += (double)(mid - begin) * 1000 / CLOCKS_PER_SEC;
  cpu_time += (double)(end - mid) * 1000 / CLOCKS_PER_SEC;
 
  printf("\nCPU processing time: %f ms\nGPU processing time: %f ms", cpu_time, gpu_time);

  // Free all our allocated memory
  cudaFree(a); cudaFree(b);
  cudaFree( c_cpu ); cudaFree( c_gpu );
}
