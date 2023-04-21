#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void organized(void)
{
	printf("thread id x: %d, thread id y: %d, thread id z: %d\n", 
		threadIdx.x, threadIdx.y, threadIdx.z);

	printf("block id x: %d, block id y: %d, block id z: %d\n", 
		blockIdx.x, blockIdx.y, blockIdx.z);


	printf("block dim x: %d, block dim y: %d, block dim z: %d\n", 
		blockDim.x, blockDim.y, blockDim.z);

	printf("grid dim x: %d, grid dim y: %d, grid dim z: %d\n", 
		gridDim.x, gridDim.y, gridDim.z);


}
int main()
{
	dim3 block(2,2);
	dim3 grid(3,3);

	organized<<<grid, block>>>();
	cudaDeviceSynchronize();

	return 0;
}