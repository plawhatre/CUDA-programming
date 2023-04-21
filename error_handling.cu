#include <stdio.h>

#define gpuError(ans) {gpuChk((ans), __FILE__, __LINE__);}
inline void gpuChk(cudaError_t error, const char *file, int line, bool abort=true)
{
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Error is: %s in file %s at line %d\n", 
			cudaGetErrorString(error), file, line);
		if (abort==true){exit(error);}
	}
	else
	{
		printf("\nError free!\n");
	}
}

__global__ void msg(int *a)
{
	int i = threadIdx.x;
	printf("index: %d, value: %d\n", i, a[i]);
}

int main()
{
	int *da;

	int a[] = {11,22,33,44};
	
	cudaError_t error;
	error = cudaMalloc(&da, 4*sizeof(int));


	if (error != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s\n", cudaGetErrorString(error));
	}
	else
	{
		printf("cool!");
	}


	gpuError(cudaMalloc(&da, 4*sizeof(int)));
	cudaDeviceSynchronize();
	return 0;
}