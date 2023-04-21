#include <stdio.h>
#include <time.h>

int seq_reduction(int *a, int size)
{
	int sum = 0;

	for (int i=0; i<size; i++)
	{
		sum += a[i];
	}
	return sum;
}

__global__ void neighboured_pair(int *a, int *tmp, int size)
{
	int tid = threadIdx.x;
	int gid = tid + blockIdx.x * blockDim.x;

	if (gid > size)
		return;

	for (int offset=1; offset <= blockDim.x/2; offset*=2 )
	{
		if (tid %(offset*2)==0)
		{
			a[gid] += a[gid+offset];
		}
		__syncthreads();
	}

	if (tid==0)
	{
		tmp[blockIdx.x] = a[gid];
	}

}

__global__ void neighboured_pair_wrap_conv(int *a, int *tmp, int size)
{
	int tid = threadIdx.x;
	int gid = tid + blockIdx.x * blockDim.x;

	int *arr = a + blockIdx.x * blockDim.x;

	if (gid > size)
		return;

	for (int offset=1; offset <= blockDim.x/2; offset*=2 )
	{
		int index = 2* offset * tid;
		if (index < blockDim.x)
		{
			arr[index] += arr[index+offset];
		}
		__syncthreads();
	}

	if (tid==0)
	{
		tmp[blockIdx.x] = a[gid];
	}

}

int main()
{
	int size=1024*50;
	int a[size];
	for (int i=0; i <size; i++){a[i] = (rand()%10);}
	// for (int i=0; i <size; i++){a[i] = 1;}
	
	// CPU implementtion
	clock_t start_cpu, end_cpu;
	double cpu_time;

	start_cpu = clock();
	int res = seq_reduction(a, size);
	end_cpu = clock();

	cpu_time = end_cpu - start_cpu;

	printf("CPU result %d\ntime taken %f\n", res, cpu_time);

	// GPU implementation

	int *da, *dtmp;

	cudaMalloc((void**)&da, size*sizeof(int));
	cudaMemcpy(da, a, size*sizeof(int), cudaMemcpyHostToDevice);

	dim3 block(128);
	dim3 grid(size/block.x);

	cudaMalloc((void**)&dtmp, grid.x*sizeof(int));
	cudaMemset(dtmp, 0, grid.x*sizeof(int));

	clock_t start_gpu, end_gpu;
	double gpu_time;

	start_gpu = clock();

	// neighboured_pair<<<grid, block>>>(da, dtmp, size);
	neighboured_pair_wrap_conv<<<grid, block>>>(da, dtmp, size);

	cudaMemcpy(a, da, size*sizeof(int), cudaMemcpyDeviceToHost);
	int tmp[grid.x];
	cudaMemcpy(tmp, dtmp, grid.x*sizeof(int), cudaMemcpyDeviceToHost);

	int sm = seq_reduction(tmp, grid.x);
	end_gpu = clock();

	gpu_time = end_gpu - start_gpu;

	printf("\n\nGPU result %d\ntime taken %f\n", sm, gpu_time);

	cudaDeviceSynchronize();

	return 0;
}