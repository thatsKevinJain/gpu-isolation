#include <iostream>
#include <math.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

using namespace cooperative_groups;

////////////////////////////////////
//				HELPERS			  //
////////////////////////////////////

#define DRIVER_API_CALL(apiFuncCall)											\
do {																			\
	CUresult _status = apiFuncCall;												\
	if (_status != CUDA_SUCCESS) {												\
		fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",	\
				__FILE__, __LINE__, #apiFuncCall, _status);						\
		exit(-1);																\
	}																			\
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)											\
do {																			\
	cudaError_t _status = apiFuncCall;											\
	if (_status != cudaSuccess) {												\
		fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",	\
				__FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));	\
		exit(-1);																\
	}																			\
} while (0)

////////////////////////////////////
//				GLOBAL			  //
////////////////////////////////////

struct Requirement {
	int func;
	int blockSize;	// blockDim
	int numBlocks;	// gridDim
	int threadIdx;
	int blockIdx;
};

////////////////////////////////////
//				KERNEL			  //
////////////////////////////////////

// Kernel function to add the elements of two arrays
__device__
void add(int n, float *y, Requirement req)
{
	int index = req.blockIdx * req.blockSize + req.threadIdx;
	int stride = req.blockSize * req.numBlocks;

	// Grid stride loop
	for (int i = index; i < n; i += stride){
		// printf("ADD: index %d, stride %d, i %d, threadIdx %d, blockIdx %d\n", index, stride, i, req.threadIdx, req.blockIdx);
		y[i] = 1.0f + y[i];
	}
}

__device__
void sub(int n, float *y, Requirement req)
{
	int index = req.blockIdx * req.blockSize + req.threadIdx;
	int stride = req.blockSize * req.numBlocks;

	// Grid stride loop
	for (int i = index; i < n; i += stride){
		// printf("SUB: index %d, stride %d, i %d, threadIdx %d, blockIdx %d\n", index, stride, i, req.threadIdx, req.blockIdx);
		y[i] = 1.0f - y[i];
	}
}

////////////////////////////////////
//			SCHEDULER			  //
////////////////////////////////////

__global__
void scheduler(int n, float *y_a1, float *y_s1, struct Requirement *queue, int length, int breadth){

	int size = length*breadth;
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	// Scheduler will simply take an item from queue and process it //
	if(threadId < length){
		for(int i=threadId; i<size; i+=length){

			// Break when no elements are left in queue //
			if(queue[i].numBlocks != NULL){
				// Call the function by using dispatcher //
				switch(queue[i].func){
					case 0: add(n,y_a1,queue[i]); break;
					case 1: sub(n,y_s1,queue[i]); break;
				}
			}
			else{
				break;
			}
		}
	}
	return;
}

////////////////////////////////////
//				MAIN			  //
////////////////////////////////////

int main(void)
{

	////////////////////////////////////
	//				CONFIG 			  //
	////////////////////////////////////

	int N = 1<<20; // array size for kernels 

	// 2048 total threads in Jetson Nano - Don't change this //
	int BLOCK_SIZE = 1024;
	int NUM_OF_BLOCKS = 2;
	
	int NUM_OF_KERNELS = 12;
	float SUM = 2.0f;
	float DIFF = 0.0f;

	////////////////////////////////////
	//			REQUIREMENTS		  //
	////////////////////////////////////
	struct Requirement *req;
	cudaMallocManaged(&req, NUM_OF_KERNELS*sizeof(Requirement (*)));

	// Fill the requirements //
	// Technically they should try to cover the whole GPU //
	for(int i=0; i<NUM_OF_KERNELS; i++){
		req[i].func = i;
		req[i].blockSize = (2048/NUM_OF_KERNELS);
		req[i].numBlocks = 1;
	}

	////////////////////////////////////
	//				INIT 			  //
	////////////////////////////////////

	float *y_A1, *y_S1;
	cudaMallocManaged(&y_A1, N*sizeof(float));
	cudaMallocManaged(&y_S1, N*sizeof(float));

	for (int i = 0; i < N; i++) {
		y_A1[i] = 1.0f;
		y_S1[i] = 1.0f;
	}

	////////////////////////////////////
	//				QUEUE			  //
	////////////////////////////////////

	// Calculate queue size //
	int length = 0; int breadth = 0;
	for(int i=0; i<NUM_OF_KERNELS; i++){
		length += req[i].blockSize;
		breadth = std::max(req[i].numBlocks, breadth);
	}
	printf("Length %d, Breadth %d\n", length, breadth);

	struct Requirement *queue;
	RUNTIME_API_CALL(cudaMallocManaged(&queue, (length*breadth*length)*sizeof(Requirement (*))));
	int cursor = 0;

	// Fill the queue as per requirements //
	for(int i=0; i<NUM_OF_KERNELS; i++){

		for(int j=0; j<req[i].numBlocks; j++){

			for(int k=0; k<req[i].blockSize; k++){
				// Create a requirement obj //
				queue[cursor].func = req[i].func;
				queue[cursor].blockSize = req[i].blockSize;
				queue[cursor].numBlocks = req[i].numBlocks;
				queue[cursor].threadIdx = k;
				queue[cursor].blockIdx = j;
				// printf("queue [%d] threadIdx[%d] blockIdx[%d] func[%d]\n", cursor, k, j, req[i].func);
				cursor++;
			}
		}
	}

	////////////////////////////////////
	//			SCHEDULER			  //
	////////////////////////////////////

	// Run the Scheduler Kernel //
	scheduler<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(N,y_A1,y_S1,queue,length,breadth);

	// Wait for GPU to finish before accessing on host //
	cudaDeviceSynchronize();

	// ERROR CHECK //
	for (int i = 0; i < N; ++i) {
		if (SUM != fabs(y_A1[i])) {
			fprintf(stderr, "add error: result verification failed\n");
			exit(-1);
		}
	}

	for (int i = 0; i < N; ++i) {
		if (DIFF != fabs(y_S1[i])) {
			fprintf(stderr, "sub error: result verification failed\n");
			exit(-1);
		}
	}

	// Free memory //
	cudaFree(y_A1);
	cudaFree(y_S1);

	return 0;
}
