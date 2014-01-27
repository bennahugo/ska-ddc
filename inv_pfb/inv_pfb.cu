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
	        printf("%d SMs are available",properties.multiProcessorCount);
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
@args stride_length number of complex numbers to process in the input array
@args d_taps *DEVICE* pointer to the filter taps which should already be preloaded on the device
@args input list of complex numbers as output by a pfb process
@args output_buffer preallocated buffer of size stride_length in which the inverse pfb output will be dumped

Note:
	The following condition must be satisfied:
	stide_length is divisable by N - all sub-windows should be completely filled
*/
void processStride(uint32_t stride_start, uint32_t stride_length, const float * d_taps, 
		   const complex_float * input, float * output_buffer){
	assert(stride_length % N == 0); //only whole blocks should be processed
	
	//Setup CU IFFT plan to process all the separate ffts in this stride in one go before we undo the filterbank
	cufftHandle plan;
	uint32_t fft_block_size = floor(N/2 + 1);
	uint32_t num_blocks = stride_length/N;
	cufftSafeCall(cufftPlan1d(&plan, fft_block_size,  CUFFT_C2R, num_blocks));
	cufftComplex * d_input;
	//there should be more than enough memory for an inplace ifft since the original data is complex. Add some padding for the filtering stage
	cudaSafeCall(cudaMalloc((void**)&d_input,sizeof(cufftComplex)*(fft_block_size*num_blocks)+PAD));
	//copy fft blocks to device where we will ifft them in batches
	//TODO: this should be changed to simply copy the spectra as is, because the pfb output of the beamformer is already only sending N/2 of each FFT
	for (uint32_t i = 0; i < stride_length; i += N)
		cudaMemcpy(d_input + (i/2 + 1) + PAD ,input + stride_start + i,sizeof(complex_float)*fft_block_size,cudaMemcpyHostToDevice); 
	//ifft the data in place (starting after the initial padding ofc)
	cufftSafeCall(cufftExecC2R(plan,d_input+PAD,(cufftReal *)(d_input+PAD)));
	cudaThreadSynchronize();	
	//reserve memory for output
	float * d_output;
	cudaSafeCall(cudaMalloc((void**)&d_output,sizeof(float)*stride_length));
	//now do the inverse pfb
	dim3 threadsPerBlock(N,1,1);
	dim3 numBlocks(stride_length / N,1,1);
	ipfb<<<numBlocks,threadsPerBlock,0>>>((float*)d_input,d_output,d_taps);
	cudaThreadSynchronize();
	cudaError code = cudaGetLastError();
	if (code != cudaSuccess){
		fprintf (stderr,"Error while executing inverse pfb -- %s\n", cudaGetErrorString(code)); 
		exit(-1);
	}
	cudaMemcpy(output_buffer, d_output,sizeof(float)*stride_length,cudaMemcpyDeviceToHost);
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
	#pragma unroll P
	for (uint32_t p = 1; p < P; ++p)
		accum = input[lB + threadIdx.x + p*N]*prototype_filter[filter_index]; //Fetching data from both the filter and the input should be coalesced
	output[lB + threadIdx.x] = accum;
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
	
	
	//Read in taps file (prototype filter)	
	float * taps = (float*) malloc(sizeof(float)*WINDOW_LENGTH);
	if (readDataFromDisk(taps_filename,sizeof(float),WINDOW_LENGTH,taps) != WINDOW_LENGTH){
		fprintf(stderr, "Prototype filter has to be %d long\n",WINDOW_LENGTH);
		return 1;
	}
	
	//Read in input file (output of earlier pfb process, so these will be complex numbers
	complex_float * pfb_data = (complex_float*) malloc(sizeof(complex_float)*(num_samples + PAD)); //pad the array with N*P positions

	if (readDataFromDisk(pfb_output_filename,sizeof(complex_float),num_samples,pfb_data + PAD) != num_samples){
		fprintf(stderr, "input pfb data does not contain %d samples\n", num_samples);
		return 1;
	}
	//Setup the device and copy the taps
	float * d_taps = initDevice(taps);
	float * output = (float*) malloc(sizeof(float)*num_samples);
	
	//do some processing
	//processStride(0, num_samples, d_taps, pfb_data, output);	
	
	//release the device
	releaseDevice(d_taps);

	//free any hostside memory:
	free(output);
	free(pfb_data);
	free(taps);	
	printf("Application Terminated Normally\n");
}
