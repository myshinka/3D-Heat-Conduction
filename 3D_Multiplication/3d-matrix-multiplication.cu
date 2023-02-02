/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

__global__ void matrixMulSeqGPU( float * c, float * a, float * b, int N )
{
  int val;
  int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
  int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
  int idx_z = threadIdx.z + blockDim.z * blockIdx.z;
  
  int stride_x = blockDim.x * gridDim.x;
  
	for( int row = idx_x; row < N; row+= stride_x )
		for( int col = idx_y; col < N; col += stride_x )
			for( int page = idx_z; page < N; page += stride_x ) {
				val = 0;
				for ( int k = 0; k < N; ++k )
					val += a[page * N + row * N + k] * b[page * N + k * N + col];
				c[page * N + row * N + col] = val;
			}
}

void matrixMulSeqCPU( float * c, float * a, float * b, int N )
{
    int val = 0;
  
	for( int row = 0; row < N; ++row ) 
		for( int col = 0; col < N; ++col ) 
			for( int page = 0; page < N; ++page ) {
				val = 0;
				for ( int k = 0; k < N; ++k ) 
					val += a[page * N + row * N + k] * b[page * N + k * N + col];				
				c[page * N + row * N + col] = val;
			}
}

void ConstantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

void MatMulPerformance(float msecTotal, int nIter, 
				   dim3 threads, dim3 dims, unsigned int mem_size)
{
  // Compute and print the performance
  float msecPerMatrixMul = msecTotal / nIter;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(dims.x) *
                             static_cast<double>(dims.y) *
							 static_cast<double>(dims.z) *
                             static_cast<double>(dims.x);
  double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  
  if(threads.x == 0)
	  printf(
		  "Performance = %.2f GFlop/s, Time = %.3f msec, Size = %.0f Ops, "
		  "TransferSize = %d kB\n\n", gigaFlops, msecTotal, 
		  flopsPerMatrixMul, (3 * mem_size) / 1024);
  else 
	  printf(
		  "Performance = %.2f GFlop/s, Time = %.3f msec, Size = %.0f Ops,\n"
		  "TransferSize = %d kB, WorkgroupSize = %u threads/block\n\n",
		  gigaFlops, msecTotal, flopsPerMatrixMul,
		  (3 * mem_size) / 1024, threads.x * threads.y * threads.z);
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char **argv,
                   int block_size, const dim3 &dims) 
{
  // Allocate host memory for matrices A and B
  //*******************************************************************
  
  unsigned int size = dims.x * dims.y * dims.z;
  unsigned int mem_size = sizeof(float) * size;
  
  float *h_A;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size));
  
  float *h_B;
  checkCudaErrors(cudaMallocHost(&h_B, mem_size));
  cudaStream_t stream;

  // Initialize host memory
  ConstantInit(h_A, size, 1.0f);
  ConstantInit(h_B, size, 0.01f);

  // Allocate device memory
  //*******************************************************************
  
  float *d_A, *d_B, *d_C;

  // Allocate host matrix C - cpu_C is for CPU, h_C for sequential GPU.
  float *h_C;
  float *cpu_C;
  checkCudaErrors(cudaMallocHost(&h_C, mem_size));
  checkCudaErrors(cudaMallocHost(&cpu_C, mem_size));

  if (h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size));
  
  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // copy host memory to device
  //*******************************************************************
  
  checkCudaErrors(
      cudaMemcpyAsync(d_A, h_A, mem_size, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(
      cudaMemcpyAsync(d_B, h_B, mem_size, cudaMemcpyHostToDevice, stream));

  // Setup execution parameters
  //*******************************************************************
  
  dim3 threads(block_size, block_size, block_size);
  dim3 grid(dims.x / threads.x, dims.y / threads.y, dims.z / threads.z);
  
  // CPU function
  //*******************************************************************
	
  printf("Computing result using CPU...\n");

  // Performs warmup operation using matrixMul CUDA kernel
  matrixMulSeqCPU(cpu_C, h_A, h_B, dims.x);

  checkCudaErrors(cudaStreamSynchronize(stream));

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, stream));

  // Execute the kernel
  int nIter = 30;

  for (int j = 0; j < nIter; j++) {
    matrixMulSeqCPU(cpu_C, h_A, h_B, dims.x);
  }

  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));
  
  printf("done\n");
  
  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  
  MatMulPerformance(msecTotal, nIter, (0,0,0), dims, mem_size);
  
  // Sequential kernel
  //*******************************************************************
  
  printf("Computing result using sequential kernel...\n");

  // Warmup operation
  matrixMulSeqGPU <<<grid, threads, 0, stream>>> (d_C, d_A, d_B, dims.x);

  checkCudaErrors(cudaStreamSynchronize(stream));

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, stream));

  // Execute the kernel
  for (int j = 0; j < nIter; j++) {
	matrixMulSeqGPU <<<grid, threads, 0, stream>>> (d_C, d_A, d_B, dims.x);
  }

  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, stream));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));
  
  printf("done\n");
  
  msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
  
  MatMulPerformance(msecTotal, nIter, threads, dims, mem_size);
  
  // Copy result from device to host
  //*******************************************************************

  checkCudaErrors(
      cudaMemcpyAsync(h_C, d_C, mem_size, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  printf("Checking computed result for correctness: ");
  bool correct = true;

  // test error
  //*******************************************************************
  
  double eps = 1.e-6;  // machine zero
  double seq_err = 0;

  for (int i = 0; i < size; i++) {
	if(seq_err > fabs(h_C[i] - cpu_C[i]))
	  seq_err = fabs(h_C[i] - cpu_C[i]);


    if (seq_err > eps) {
      printf("Error! Sequential Matrix[%05d]=%.8f, error term is > %E\n",
             i, h_C[i], eps);
      correct = false;
    }
  }

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");


  // Clean up memory
  //*******************************************************************
  
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C));
  checkCudaErrors(cudaFreeHost(cpu_C));
  
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  if (correct) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}


/***************************************************************************
 * Program main
 ***************************************************************************/
 
int main(int argc, char **argv) {
  printf("[Matrix Multiply Using CUDA] - Starting...\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    printf("      -N=MatSize (Dimensions (N * N * N) of matrices)\n");

    exit(EXIT_SUCCESS);
  }

  // This will pick the best possible CUDA capable device, otherwise
  // override the device ID based on input provided at the command line
  int dev = findCudaDevice(argc, (const char **)argv);

  int block_size = 32;

  dim3 dims(block_size, block_size, block_size);

  // Dimentions of matrices
  if (checkCmdLineFlag(argc, (const char **)argv, "N")) {
    dims.x = dims.y = dims.z = getCmdLineArgumentInt(argc, (const char **)argv, "N");
  }

  printf("Matrices(%d,%d,%d)\n", dims.x, dims.y, dims.z);

  checkCudaErrors(cudaProfilerStart());
  int matrix_result = MatrixMultiply(argc, argv, block_size, dims);
  checkCudaErrors(cudaProfilerStop());

  exit(matrix_result);
}
