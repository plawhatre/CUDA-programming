#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void get_parameters()
{
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount >= 1)
	{
		printf("Number of cuda device found is %d\n", deviceCount);
	}
	else{printf("No device found!");}

	int dno = 0;
	cudaDeviceProp iProp;

	cudaGetDeviceProperties(&iProp, dno);

	// Name of the device
	printf("Name: %s\n", iProp.name);

	// Major compute capability
	printf("Major Compute Capability: %d\n", iProp.major);

	// Minor compute capability
	printf("Major Compute Capability: %d\n", iProp.minor);

	// Total multiprocessor count
	printf("Multiprocessor count: %d\n",iProp.multiProcessorCount);

	// Total Global Memory
	printf("Total global Memory(in MB): %zu\n",((iProp.totalGlobalMem/1024)/1024));

	// Total Constant Memory
	printf("Total constant Memory(in KB): %zu\n",(iProp.totalConstMem/1024));

	// Shared Memory per Block
	printf("shared memory per block (in KB): %zu\n",(iProp.sharedMemPerBlock/1024));

	// Maximum threads per block
	printf("Maximum thread per block: %d\n",iProp.maxThreadsPerBlock);

	// Maximum Thread Dimensions 
	printf("Maximum thread dimensions: (%d, %d, %d)\n", 
		iProp.maxThreadsDim[0],
		iProp.maxThreadsDim[1],
		iProp.maxThreadsDim[2]);

	// Maximum Grid Size
	printf("Maximum Grid Size: (%d, %d, %d)\n",
		iProp.maxGridSize[0],
		iProp.maxGridSize[1],
		iProp.maxGridSize[2]);

	// Clock rate 
	printf("Clock rate: %d kHz\n", iProp.clockRate);

	// Wrap Size
	printf("Warp Size: %d\n", iProp.warpSize);


}

int main()
{
	get_parameters();
	cudaDeviceSynchronize();
	return 0;
}