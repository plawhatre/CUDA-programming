#include <stdio.h>

__global__ void add(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int main()
{
	int size = 4;

	const int a[] = {1, 2, 3, 4};
	const int b[size] = {100, 200, 300, 400};
	int c[size] = {0,0,0,0};

	int *da, *db, *dc;

	cudaMalloc((void **) & da, size*sizeof(int));
	cudaMalloc((void **) & db, size*sizeof(int));
	cudaMalloc((void **) & dc, size*sizeof(int));

	cudaMemcpy(da, a, size*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, size*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dc, c, size*sizeof(int),cudaMemcpyHostToDevice);


	add <<<1, size>>>(dc, da, db);

	cudaMemcpy(c, dc, size*sizeof(int), cudaMemcpyDeviceToHost);


	cudaDeviceSynchronize();
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	for (int x=0; x<size; x++)
	{
		printf("%d\n", c[x]);
	}


	return 0;


}