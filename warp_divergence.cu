#include <stdio.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void non_divergent()
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int a;

	if (tid/32 % 2 == 0)
	{
		a = 10;
		printf("index: %d, val: %d\n", tid);
	}
	else
	{    
		a = 20;
		printf("index: %d, val: %d\n", tid);
	}
}

__global__ void divergent()
{
	int tid = threadIdx.x;
	int a;

	if (tid % 2 != 0)
	{
		a = 10;
		printf("d_index: %d, val: %d\n", tid);
	}
	else
	{
		a = 20;
		printf("d_index: %d, val: %d\n", tid);
	}
}


int main()
{
	non_divergent<<<1000,32>>>();
	divergent<<<1000,32>>>();

	cudaDeviceSynchronize();
	return 0;
}