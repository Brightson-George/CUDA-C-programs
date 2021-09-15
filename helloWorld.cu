#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

__global__ void helloWorld() {
	printf("Hello World\n");
}

int main()
{
	int nx, ny;
	nx = 16;
	ny = 4;
	
	dim3 block(8, 2, 1);
	dim3 grid(nx / block.x, ny / block.y, 1);
	
	helloWorld << <grid, block >> > ();
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}