#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufft.h>
#include <cutil_inline.h>
#include <assert.h>
#include "math.h"
#include "file_reader.h"

#define N 512 //Number of FFT samples
#define P 8 //Number of Filterbanks
#define WINDOW_LENGTH N*P
#define PAD N*P

typedef struct {
	float r;
	float i;
} complex_float;

/****************************************************************************************************************
 Forward declarations
*****************************************************************************************************************/

void copyTaps(const float * taps, float * d_taps);
void processStride(uint32_t stride_start, uint32_t stride_length, const float * d_taps,
                   const complex_float * input);
__global__ void ipfb(float * input,  float * output, float * prototype_filter);


void copyTaps(const float * taps, float * d_taps){
	//TODO: COPY TAPS TO DEVICE CONSTANT MEMORY
}


void processStride(uint32_t stride_start, uint32_t stride_length, const float * d_taps, 
		   const complex_float * input){
	//Setup CU FFT plan
	assert(stride_length % N == 0); //only whole blocks should be processed
	cufftHandle plan;
	uint32_t fft_block_size = floor(N/2 + 1);
	uint32_t num_blocks = stride_length/N;
	cufftSafeCall(cufftPlan1d(plan, fft_block_size,  CUFFT_C2R, num_blocks));
	cudaSafeCall(cudaMemcpy());
	cufftComplex * d_input;
	//there should be more than enough memory for an inplace ifft since the original data is complex. Add some padding for the filtering stage
	cudaSafeCall(cudaMalloc(void**)&d_input,sizeof(cufftComplex)*(fft_block_size*num_blocks)+PAD);
	//copy fft blocks to device where we will ifft them in batches
	for (uint32_t i = 0; i < stride_length; i += ifft_block_size)
		cudaMemcpy(d_input + i + PAD, input + stride_start + i,sizeof(complex_float)*ifft_block_size,cudaMemcpyHostToDevice); 
	//ifft the data in place (starting after the initial padding ofc)
	cufftSafeCall(cufftExecC2R(plan,d_input+PAD,(cufftReaxl *)(d_input+PAD)));
	cufftSafeCall(cufftDestroy(plan));
	cudaThreadSyncronize();
	
	cudaFree(d_input);
	cufftSafeCall(cufftDestroy(plan));
}

/**
This kernel computes the inverse polyphase filter bank operation. 
The kernel should be invoked with blockDim = N and numBlocks = (stride_length / N)

It will perform the following filterbank algorithm:
for (l = 0; l < stride_length; l += N) in parallel
	for (n = 0; n < N; ++n) in parallel
		accum = x[l+n]h[N - n - 1]
		for (p = 1; p < P; ++p)
			accum = x[l+n+p*N]h[p*N + (N - n - 1)]
		endfor
	endfor
endfor

Technically the prototype filter remains the same for the synthesis filter bank. Each subfilter, however has to be read in reverse. Furthermore
the commutator technically runs in reverse as well, meaning that we should flip the order subfilters are executed in the bank. But whether we accumulate
each y[n] in forward or reverse does not matter. It should be clear that if the 3rd loop is run backwards with the the initialization of the 
accumulation set to a position in the last subfilter the result should remain the same.
*/
__global__ void ipfb(float * input,  float * output, float * prototype_filter){
	uint32_t lB = blockIdx.x * blockDim.x;
	uint32_t filter_index = N-n-1;
	register float accum = input[lB + threadIdx.x]*prototype_filter[filter_index]; //Fetching data from both the filter and the input should be coalesced
	#pragma unroll P
	for (p = 1; p < P; ++p)
		accum = input[lB + threadIdx.x + p*N]*prototype_filter[filter_index]; //Fetching data from both the filter and the input should be coalesced
}

int main ( int argc, char ** argv ){
	char * taps_filename;
	char * pfb_output_filename;
	char * output_filename;
	uint32_t num_samples;
	if (argc != 5){
		fprintf(stderr, "expected arguements: 'prototype_filter_file' 'no_complex_to_read' 'pfb_output_file' 'output_file'\n");
		return 1;
	}	
	taps_filename = argv[1];
	num_samples = atoi(argv[2]);
	pfb_output_filename = argv[3];
	output_filename = argv[4];
	
	/****************************************************************************************************************
	 Read in taps file (prototype filter)
	*****************************************************************************************************************/
	float * taps = (float*) malloc(sizeof(float)*WINDOW_LENGTH);
	if (readDataFromDisk(taps_filename,sizeof(float),WINDOW_LENGTH,taps) != WINDOW_LENGTH){
		fprintf(stderr, "Prototype filter has to be %d long\n",WINDOW_LENGTH);
		return 1;
	}
	/****************************************************************************************************************
	 Read in input file (output of earlier pfb process, so these will be complex numbers
        *****************************************************************************************************************/
	complex_float * pfb_data = (complex_float*) malloc(sizeof(complex_float)*(num_samples + PAD)); //pad the array with N*P positions

	if (readDataFromDisk(pfb_output_filename,sizeof(complex_float),num_samples,pfb_data + PAD) != num_samples){
		fprintf(stderr, "input pfb data does not contain the %d of samples\n", num_samples);
		return 1;
	}
	processStride(0, num_samples, NULL,pfb_data);	

	free(pfb_data);
	free(taps);	
}
