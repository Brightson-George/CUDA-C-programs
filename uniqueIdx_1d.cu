#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>


__global__ void unique_idx_calc_threadIdx(int * input) {
	int tid = threadIdx.x;
	printf("threadIdx_X: %d, value: %d \n", tid, input[tid]);
}

__global__ void unique_gid_calculation(int* input) {
	int tid = threadIdx.x;
	int offset = blockDim.x * blockIdx.x;
	int gid = offset + threadIdx.x;
	printf("threadIdx_X: %d, globalIdx_X: %d, value: %d \n", tid, gid, input[gid]);
}

int main() {
	int array_size = 8;
	int array_byte_size = sizeof(int) * array_size;

	int h_data[] = { 1, 2, 3, 4, 5, 6, 7, 8 };

	for (int i = 0; i < array_size; i++) {
		printf(" %d ", h_data[i]);
	}
	printf("\n \n");

	int * d_data;
	cudaMalloc(((void**)&d_data), array_byte_size);
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(4, 1, 1);
	dim3 grid(4, 1, 1);

	//unique_idx_calc_threadIdx << <grid, block >> > (d_data);
	unique_gid_calculation << < grid, block >> > (d_data);
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;

}

