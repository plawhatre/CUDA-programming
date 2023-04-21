#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void access(int *arr)
{
	int i = threadIdx.x;
	printf("index: %d, value: %d \n", i, arr[i]);
}

__global__ void offset_access_1d_block_1d_grid(int *arr)
{
	int block_offset = blockIdx.x * blockDim.x;
	int i = threadIdx.x + block_offset;
	printf("index: %d, value: %d\n", 
		i, arr[i]);
}

__global__ void offset_access_1d_block_2d_grid(int *arr)
{
	int row_offset = blockDim.x * gridDim.x * blockIdx.y ;
	int block_offset = blockIdx.x * blockDim.x;
	int i = threadIdx.x + block_offset + row_offset;
	printf("index: %d, value: %d\n", 
		i, arr[i]);
}

__global__ void offset_access_2d_block_2d_grid(int *arr)
{	
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	int block_offset = blockIdx.x * blockDim.x * blockDim.y;
	int row_offset = blockIdx.y * gridDim.x * blockDim.x * blockDim.y;

	int i = tid + block_offset + row_offset;
	
	printf("index: %d, value: %d\n", 
		i, arr[i]);
}

int main()
{
	int *a;
	int size = 16;
	int arr[size] = {-1,-2,-3, -4, -5, -6, -7, -8,
		-9, -10, -11, -12, -13, -14, -15, -16};

	cudaMalloc(&a, size*sizeof(int));

	cudaMemcpy(a, arr, size*sizeof(int), cudaMemcpyHostToDevice);

	// 1. one element per thread
	// dim3 block(size);
	// dim3 grid(1);
	// access <<<grid, block>>>(a);

	// 2. elements from 1D block of 1D grid
	// dim3 block(4);
	// dim3 grid(4);
	// offset_access_1d_block_1d_grid <<<grid, block>>>(a);

	// 3. elements from 1D block of 2D grid
	// dim3 block(4);
	// dim3 grid(2,2);
	// offset_access_1d_block_2d_grid <<<grid, block>>>(a);

	// 4. elements from 1D block of 2D grid
	dim3 block(2,2);
	dim3 grid(2,2);
	offset_access_2d_block_2d_grid <<<grid, block>>>(a);

	cudaDeviceSynchronize();

	return 0;
}