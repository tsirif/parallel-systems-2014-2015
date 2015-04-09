#include <cuda_runtime.h>
#include <stdio.h>

int main()
{
    cudaDeviceProp props;
    int devCount;
    cudaGetDeviceCount(&devCount);
    
    for(int i = 0; i < devCount; ++i)
    {
            cudaGetDeviceProperties(&props, 0);
            printf("%d%d\n", props.major, props.minor);
    }
}
