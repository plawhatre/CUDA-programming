#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello(void)
{
	printf("hello cuda \n");
}

int main()
{
	dim3 block(4,2);
	dim3 grid(8/block.x,16/block.y);

	hello<<<grid, block>>>();
	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}