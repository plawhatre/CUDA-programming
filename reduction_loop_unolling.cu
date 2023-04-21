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

__global__ void loop_unrolling_block2(int *a, int *tmp, int size)
{
	int tid = threadIdx.x;

	int BLOCK_OFFSET = blockIdx.x * blockDim.x * 2;

	int *a_new = a + BLOCK_OFFSET;
	int gid = tid + BLOCK_OFFSET;

	if ((gid+blockDim.x) < size)
	{
		a[gid] += a[gid+blockDim.x];
	}

	__syncthreads();

	for (int offset=blockDim.x/2; offset > 0; offset/=2 )
	{
		if (tid < offset)
		{
			a_new[tid] += a_new[tid+offset];
		}
		__syncthreads();
	}

	if (tid==0)
	{
		tmp[blockIdx.x] = a_new[0];
	}

}

__global__ void loop_unrolling_block4(int *a, int *tmp, int size)
{
	int tid = threadIdx.x;

	int BLOCK_OFFSET = blockIdx.x * blockDim.x * 4;

	int *a_new = a + BLOCK_OFFSET;
	int gid = tid + BLOCK_OFFSET;

	if ((gid+3*blockDim.x) < size)
	{
		int a1 = a[gid];
		int a2 = a[gid+1*blockDim.x];
		int a3 = a[gid+2*blockDim.x];
		int a4 = a[gid+3*blockDim.x];
		a[gid] = a1+a2+a3+a4;
	}

	__syncthreads();

	for (int offset=blockDim.x/2; offset > 0; offset/=2 )
	{
		if (tid < offset)
		{
			a_new[tid] += a_new[tid+offset];
		}
		__syncthreads();
	}

	if (tid==0)
	{
		tmp[blockIdx.x] = a_new[0];
	}

}

int main()
{
	int size=1024*50;
	int a[size];
	// for (int i=0; i <size; i++){a[i] = (rand()%10);}
	for (int i=0; i <size; i++){a[i] = 1;}

	// GPU implementation

	int *da, *dtmp;

	cudaMalloc((void**)&da, size*sizeof(int));
	cudaMemcpy(da, a, size*sizeof(int), cudaMemcpyHostToDevice);

	dim3 block(256);
	dim3 grid((size/block.x)/4);

	cudaMalloc((void**)&dtmp, grid.x*sizeof(int));
	cudaMemset(dtmp, 0, grid.x*sizeof(int));

	clock_t start_gpu, end_gpu;
	double gpu_time;

	start_gpu = clock();

	loop_unrolling_block4<<<grid, block>>>(da, dtmp, size);

	cudaMemcpy(a, da, size*sizeof(int), cudaMemcpyDeviceToHost);
	int tmp[grid.x];
	cudaMemcpy(tmp, dtmp, grid.x*sizeof(int), cudaMemcpyDeviceToHost);

	int sm = seq_reduction(tmp, grid.x);
	end_gpu = clock();

	gpu_time = end_gpu - start_gpu;

	printf("GPU result %d\ntime taken %f\n", sm, gpu_time);

	cudaDeviceSynchronize();

	return 0;
}