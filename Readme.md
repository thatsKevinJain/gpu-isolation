# EdgeIso
Edge computing is gaining a lot of traction due to its ability to process large amounts of data at the edge of the network thus reducing workloads at the cloud. They also improve response times for AI applications leading to better user experience. Edge devices find many applications today ranging from autonomous vehicles, cloud gaming, content delivery & real time traffic management. However, they are limited in terms of resources (CPU, GPU and memory) and thus maintaining QoS while running multiple AI applications becomes difficult. We investigate latency issues introduced by running multiple applications on edge devices and propose a scheduler that aims to break the workloads into batches of smaller tasks and queue them on the GPUs so that they run in limited space. It aims to isolate the processes from each other and mitigate resource interference. Our scheduler will effectively allow maximizing the throughput of GPUs and also enable guarantees in reponse time.

Check `edge-iso.pdf` for in-depth explanation. Our work is insipired from this paper - [Fractional GPUs](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8743200)

### Code explanation

#### Default solution
- Launch multiple kernels parallely using `streams`

```c++
function add(int *y){
	i = threadId;
	y[i] = 1.0f + y[i];
	...
}

function sub(int *y){
	i = threadId;
	y[i] = 1.0f - y[i];
	...
}

function main(){

	// Define & initialize variables //
	float *y;
	cudaMallocManaged(&y, N*sizeof(float));
	...

	// Create streams //
	for (int i = 0; i < NO_OF_KERNELS; ++i){
	    cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
	}
	...

	// Launch the kernels parallely //
	add<<<numBlocks, blockSize, 0, stream[0]>>>(N,y);
	subtract<<<numBlocks, blockSize, 0, stream[1]>>>(N,y);
	...

	// Synchronize kernels //
	cudaDeviceSynchronize();

	// Cleanup //
	cudaFree(y);
}
```

#### EdgeIso
- Launch one kernel that smartly does the work of all other kernels. **This can be ideal only when you preemptively have the knowledge of all kernels that will be run in the future, it won't work with kernels that process data of different shapes and sizes**

```c++
struct Requirement {
	int func;
	int blockSize;	// blockDim
	int numBlocks;	// gridDim
	int threadIdx;
	int blockIdx;
};

function add(int *y, Requirement req){
	i = req.threadId;
	y[i] = 1.0f + y[i];
	...
}

function sub(int *y, Requirement req){
	i = req.threadId;
	y[i] = 1.0f - y[i];
	...
}

function scheduler(float *y, struct Requirement *queue){

	// Scheduler will simply take an item from queue and process it //
	while(<condition>){
		// Call the function by using dispatcher //
		switch(queue[i].func){
			case 0: add(y,queue[i]); break;
			case 1: sub(y,queue[i]); break;
		}
		...
	}
	...
}

function main(){

	// Define & initialize variables //
	float *y;
	cudaMallocManaged(&y, N*sizeof(float));
	...

	// Fill the requirements //
	struct Requirement *req;
	for (int i = 0; i < NO_OF_KERNELS; ++i){
		req[i].func = i;
		req[i].blockSize = (2048/NUM_OF_KERNELS);
		req[i].numBlocks = 1;
	}
	...

	// Create queues using mapping //
	// Technically we are just adjusting thread and block numbers to process them as per our new partitions //

	for(int i=0; i<NUM_OF_KERNELS; i++){

		for(int j=0; j<req[i].numBlocks; j++){

			for(int k=0; k<req[i].blockSize; k++){
				// Create a requirement obj //
				queue[cursor].func = req[i].func;
				queue[cursor].blockSize = req[i].blockSize;
				queue[cursor].numBlocks = req[i].numBlocks;
				queue[cursor].threadIdx = k;
				queue[cursor].blockIdx = j;

				cursor++;
			}
		}
	}
	... 

	// Launch the scheduler //
	scheduler<<<NUM_OF_BLOCKS, BLOCK_SIZE>>>(y, queue);
	...

	// Synchronize kernels //
	cudaDeviceSynchronize();

	// Cleanup //
	cudaFree(y);
}
```

This solution tries runs faster the default solution for the same workload with a huge factor, check `edge-iso.pdf` for performance improvements.

---
