#include <stdio.h>

__inline__ __device__ unsigned long long __warpReduce(unsigned long long sum){
	sum += __shfl_xor_sync(0xffffffff, sum, 1);
	sum += __shfl_xor_sync(0xffffffff, sum, 2);
	sum += __shfl_xor_sync(0xffffffff, sum, 4);
	sum += __shfl_xor_sync(0xffffffff, sum, 8);
	sum += __shfl_xor_sync(0xffffffff, sum, 16);
	return sum;
}

__global__ void __sievePrime(bool* tempStore, const unsigned long total){
	unsigned long idx = (threadIdx.x + blockIdx.x * blockDim.x << 1) + 3;
	for(unsigned long i = idx * idx; i <= (total + 1) << 1; i += idx << 4){
		tempStore[(i - 3) >> 1] = false;
		tempStore[((idx << 1) + i - 3) >> 1] = false;
		tempStore[((idx << 2) + i - 3) >> 1] = false;
		tempStore[(6 * idx + i - 3) >> 1] = false;
		tempStore[((idx << 3) + i - 3) >> 1] = false;
		tempStore[(10 * idx + i - 3) >> 1] = false;
		tempStore[(12 * idx + i - 3) >> 1] = false;
		tempStore[(14 * idx + i - 3) >> 1] = false;
	}
}

__global__ void __parallelSum(bool* tempStore, const unsigned long total, unsigned long long* primeSum){
	unsigned long idx = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ unsigned char warpSum[32];
	unsigned char warpID = threadIdx.x / warpSize, laneID = threadIdx.x % warpSize; 
	unsigned long long sum = (idx < total)? tempStore[idx] : 0;
	sum = __warpReduce(sum);
	if (laneID == 0) warpSum[warpID] = sum;
       __syncthreads();
       sum = (threadIdx.x < warpSize)? warpSum[laneID] : 0;
       if (warpID == 0) sum = __warpReduce(sum);
       if (threadIdx.x == 0) atomicAdd(primeSum, sum);
}

unsigned long long prime(unsigned long long total){
	bool *tempStore;
	total = total / 2 - 1;
	unsigned long long *devPrimeSum, *hostPrimeSum;
	unsigned long launch = sqrt(total);
	hostPrimeSum = (unsigned long long*)malloc(sizeof(unsigned long long));
	cudaMalloc(&devPrimeSum, sizeof(unsigned long long));
	cudaMalloc(&tempStore, total * sizeof(bool));
	cudaMemset(tempStore, true, total * sizeof(bool));
	__sievePrime <<< launch, 1 >>> (tempStore, total);
	cudaDeviceSynchronize();
	__parallelSum <<< total / 128 + 1, 128 >>> (tempStore, total, devPrimeSum);
	cudaMemcpy(hostPrimeSum, devPrimeSum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaFree(tempStore);
	cudaFree(devPrimeSum);
	return *hostPrimeSum + 1;
}

int main(){
	printf("Total primes for 1 billion numbers -> %d\n", prime(1000000000));
	return 0;
}
