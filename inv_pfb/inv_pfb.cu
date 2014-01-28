#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufft.h>
#include <assert.h>
#include "math.h"
#include "file_reader.h"

#define N 512 //Number of FFT samples
#define P 8 //Number of Filterbanks
#define WINDOW_LENGTH N*P
#define PAD N*P

/****************************************************************************************************************
 debugging flags
*****************************************************************************************************************/
#define DUMP_IFFT_DATA_TO_DISK //uncomment to dump ifft data (of last processed stride) to disk
#define IFFT_DATA_OUTPUT_FILE "/home/bhugo/ska-res/ska-ddc/inv_pfb/FFT_INV_ON_PFB_DATA.dat"
#define DUMP_TRIMMED_DATA
#define TRIMMED_DATA_OUTPUT_FILE "/home/bhugo/ska-res/ska-ddc/inv_pfb/TRIMMED_PFB.dat"

typedef struct {
	float r;
	float i;
} complex_float;

/****************************************************************************************************************
 Forward declarations
*****************************************************************************************************************/

float * initDevice(const float * taps);
void releaseDevice(const float * d_taps);
void processStride(uint32_t stride_start, uint32_t stride_length, const float * d_taps,
                   const complex_float * input, float * output_buffer);
__global__ void ipfb(const float * input,  float * output, const float * prototype_filter);

/****************************************************************************************************************
Error handling macros
****************************************************************************************************************/
#define cudaSafeCall(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

inline void __cufftSafeCall( uint32_t err, const char *file, const int line ){
        if ( CUFFT_SUCCESS != err ){
                fprintf( stderr, "cufftSafeCall() failed at %s:%i\n", file, line);
                exit( -1 );
        }
        return;
}
#define cufftSafeCall(err)  __cufftSafeCall(err, __FILE__, __LINE__)

float * initDevice(const float * taps){
	//Choose a reasonably good device(https://www.cs.virginia.edu/~csadmin/wiki/index.php/CUDA_Support/Choosing_a_GPU), based on # SMs:
	int num_devices, device;
	cudaGetDeviceCount(&num_devices);
	if (num_devices > 0) {
		//get the argmax{devID}(multiProcessorCounts):
      		int max_multiprocessors = 0, max_device = 0;
      		for (device = 0; device < num_devices; device++) {
              		cudaDeviceProp properties;
	              	cudaGetDeviceProperties(&properties, device);
			if (max_multiprocessors < properties.multiProcessorCount) {
         	             max_multiprocessors = properties.multiProcessorCount;
                	     max_device = device;
			}
		}
		cudaSetDevice(max_device);
        	cudaDeviceReset();
		
		//print some stats:
        	cudaDeviceProp properties;
	        cudaGetDeviceProperties(&properties, max_device);

        	size_t mem_tot = 0;
	        size_t mem_free = 0;
	        cudaMemGetInfo  (&mem_free, & mem_tot);
        	printf("Chosen GPU Statistics\n---------------------------------------------------------\n");
	        printf("%s, device %d on PCI Bus #%d, clocked at %f GHz\n",properties.name,properties.pciDeviceID,
        	        properties.pciBusID,properties.clockRate / 1000000.0);
	        printf("Compute capability %d.%d with %f GiB global memory (%f GiB free)\n",properties.major,
        	        properties.minor,mem_tot/1024.0/1024.0/1024.0,mem_free/1024.0/1024.0/1024.0);
	        printf("%d SMs are available\n",properties.multiProcessorCount);
	        printf("---------------------------------------------------------\n");
	} else {
		fprintf(stderr,"Cannot find suitable GPU device. Giving up");
		exit(-1);
	}


	/*For now we'll copy the taps into global memory. It doesn't make sense to copy this to constant memory
	  as the individual threads in each warp will be accessing different locations, and will therefore be serialized.
	  Instead memory calls should be coalesced for each warp of threads due to the nice accessing pattern of the
	  Weighted Window Overlap Add method. One optimization trick that one may try is to copy this into texture memory
	  where there may be a slight performance increase due to the texture caching properties of the GPU.
	*/
	float * d_taps;
	cudaSafeCall(cudaMalloc((void**)&d_taps,sizeof(float)*WINDOW_LENGTH));
	cudaSafeCall(cudaMemcpy(d_taps,taps,sizeof(float)*WINDOW_LENGTH,cudaMemcpyHostToDevice));
	
	return d_taps;
}

void releaseDevice(float * d_taps){
	cudaSafeCall(cudaFree(d_taps));
	cudaDeviceReset(); //leave the device in a safe state
}

/**
This method will process a subset of the complex input from a pfb process and output the inverse pfb.
@args stride_start starting position in input array
@args stride_length_in_blocks number of N/2 + 1 length FFTs contained in the input
@args d_taps *DEVICE* pointer to the filter taps which should already be preloaded on the device
@args input list of complex numbers as output by a pfb process
@args output_buffer preallocated buffer of size stride_length in which the inverse pfb output will be dumped

Note:
	The following condition must be satisfied:
	length of the input should be divisiable by the block length (2 / N + 1) - as it should be when output by a pfb 
*/
void processStride(uint32_t stride_start, uint32_t stride_length_in_blocks, const float * d_taps, 
		   const complex_float * input, float * output_buffer){
	uint32_t fft_block_size = N/2 + 1;
	//Setup CU IFFT plan to process all the separate ffts in this stride in one go before we undo the filterbank
	cufftHandle plan;
	cufftSafeCall(cufftPlan1d(&plan, N,  CUFFT_C2R, stride_length_in_blocks));
	cufftComplex * d_input;
	//there should be more than enough memory for an inplace ifft since the original data is complex. Add some padding for the filtering stage:
	cudaSafeCall(cudaMalloc((void**)&d_input,sizeof(cufftComplex) * fft_block_size * stride_length_in_blocks + sizeof(cufftReal) * PAD));	
	//copy everything into the device input vector, after the initial padding (in floats)
	cudaSafeCall(cudaMemcpy((cufftReal *)d_input + PAD,input,sizeof(complex_float) * stride_length_in_blocks * fft_block_size,cudaMemcpyHostToDevice));
	//ifft the data in place (starting after the initial padding ofc)
	cufftSafeCall(cufftExecC2R(plan,(cufftComplex *)((cufftReal *)d_input + PAD),(cufftReal *)d_input + PAD));
	cudaThreadSynchronize();	
	//reserve memory for output
	float * d_output;
	uint32_t output_length_in_samples = stride_length_in_blocks * N;
	cudaSafeCall(cudaMalloc((void**)&d_output,sizeof(float)*output_length_in_samples));
	//dump ifft data to disk for debugging
        #ifdef DUMP_IFFT_DATA_TO_DISK
                float * ifft_out = (float *)malloc(sizeof(float)*output_length_in_samples);
                cudaSafeCall(cudaMemcpy(ifft_out,(cufftReal *)d_input + PAD, sizeof(cufftReal) * output_length_in_samples,cudaMemcpyDeviceToHost));
                writeDataToDisk(IFFT_DATA_OUTPUT_FILE,sizeof(float),output_length_in_samples,ifft_out);
                free(ifft_out);
        #endif
	/*
	//now do the inverse pfb
	dim3 threadsPerBlock(N,1,1);
	dim3 numBlocks(stride_length_in_blocks,1,1);
	ipfb<<<numBlocks,threadsPerBlock,0>>>((float*)d_input,d_output,d_taps);
	cudaThreadSynchronize();
	cudaError code = cudaGetLastError();
	if (code != cudaSuccess){
		fprintf (stderr,"Error while executing inverse pfb -- %s\n", cudaGetErrorString(code)); 
		exit(-1);
	}
	cudaMemcpy(output_buffer, d_output,sizeof(float)*output_length_in_samples,cudaMemcpyDeviceToHost);
	*/
	//finally free device memory and destroy the IFFT plan
	cudaFree(d_input);
	cudaFree(d_output);	
	cufftSafeCall(cufftDestroy(plan));
}

/**
This kernel computes the inverse polyphase filter bank operation (Weighted Window Overlap Add method). 

Background theory:
	For the forward process see:
		https://casper.berkeley.edu/wiki/The_Polyphase_Filter_Bank_Technique
	The inverse process has its subfilters flipped, but follows the same trend, see:
		A REVIEW OF POLYPHASE FILTER BANKS AND THEIR APPLICATION --- Daniel Zhou. 
		Air Force Research Laboratory. AFRL-IF-RS-TR-2006-277 In-House Final 
		Technical Report, September 2006.

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
__global__ void ipfb(const float * input,  float * output, const float * prototype_filter){
	uint32_t lB = blockIdx.x * blockDim.x;
	uint32_t filter_index = N-threadIdx.x-1;
	register float accum = input[lB + threadIdx.x]*prototype_filter[filter_index]; //Fetching data from both the filter and the input should be coalesced
	#pragma unroll
	for (uint32_t p = 1; p < P; ++p)
		accum = input[lB + threadIdx.x + p*N]*prototype_filter[filter_index]; //Fetching data from both the filter and the input should be coalesced
	output[lB + threadIdx.x] = accum;
}

void cutDuplicatesFromData(const complex_float * in, complex_float * out, uint32_t orig_count){
	assert(orig_count % N == 0); //only whole blocks should be processed
	for (uint32_t i = 0; i < orig_count; i += N){
		memcpy(out + (i/N)*(N/2 + 1), in + i, (N/2 + 1) * sizeof(complex_float));
	}
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
	printf("Performing operation on %d samples from '%s'\n",num_samples,pfb_output_filename);
	
	//Read in taps file (prototype filter)	
	float * taps = (float*) malloc(sizeof(float)*WINDOW_LENGTH);
	if (readDataFromDisk(taps_filename,sizeof(float),WINDOW_LENGTH,taps) != WINDOW_LENGTH){
		fprintf(stderr, "Prototype filter has to be %d long\n",WINDOW_LENGTH);
		return 1;
	}
	
	//Read in input file (output of earlier pfb process, so these will be complex numbers
	complex_float * pfb_data = (complex_float*) malloc(sizeof(complex_float)*(num_samples)); //pad the array with N*P positions

	if (readDataFromDisk(pfb_output_filename,sizeof(complex_float),num_samples,pfb_data) != num_samples){
		fprintf(stderr, "input pfb data does not contain %d samples\n", num_samples);
		return 1;
	}
	//Format the data into N/2 + 1 sized chunks to cut away unnecessary samples for the IFFT
	uint32_t trimmed_input_length = (N/2+1)*(num_samples / N);
	complex_float * trimmed_pfb_data = (complex_float*) malloc(sizeof(complex_float)*trimmed_input_length);
	cutDuplicatesFromData(pfb_data,trimmed_pfb_data,num_samples);
	#ifdef DUMP_TRIMMED_DATA
		writeDataToDisk(TRIMMED_DATA_OUTPUT_FILE,sizeof(complex_float),trimmed_input_length,trimmed_pfb_data);
	#endif
	//Setup the device and copy the taps
	float * d_taps = initDevice(taps);
	float * output = (float*) malloc(sizeof(float)*num_samples);
	
	//do some processing
	processStride(0, num_samples/N, d_taps, trimmed_pfb_data, output);	
        writeDataToDisk(output_filename,sizeof(float),num_samples,output);
	
	//release the device
	releaseDevice(d_taps);

	//free any hostside memory:
	free(output);
	free(pfb_data);
	free(taps);
	free(trimmed_pfb_data);	
	printf("Application Terminated Normally\n");
}
