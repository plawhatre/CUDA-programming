#include <stdio.h>

__global__ void add(int a, int b, int *c)
{
    *c = a + b;
}

int main()
{
    int c ,* devc ;
    cudaMalloc((void **) & devc , sizeof ( int ) ) ;
    
    add <<<1 ,1 >>>(2 ,5 , devc ) ;
    cudaMemcpy (& c , devc , sizeof ( int ) , cudaMemcpyDeviceToHost) ;
    
    cudaFree(devc) ;
    
    printf("2+5 = %d \n" , c ) ;
    return 0;
}