/**
  _____ _   ___      ________ _____   _____ ______   _____  ______ ____  
 |_   _| \ | \ \    / /  ____|  __ \ / ____|  ____| |  __ \|  ____|  _ \ 
   | | |  \| |\ \  / /| |__  | |__) | (___ | |__    | |__) | |__  | |_) |
   | | | . ` | \ \/ / |  __| |  _  / \___ \|  __|   |  ___/|  __| |  _ < 
  _| |_| |\  |  \  /  | |____| | \ \ ____) | |____  | |    | |    | |_) |
 |_____|_| \_|   \/   |______|_|  \_\_____/|______| |_|    |_|    |____/ 
                                                                         
This is a CUDA implementation of the inverse Polyphase Filter Bank (PFB), also known as the Weighted 
Overlap Add Method. It constructs a basic synthesis filterbank that can process output in strides and 
is therefore suitable for real time operation where the input length is not known in advance. Note 
that this filterbank construction does not provide Perfect Reconstruction (PR) of the original input 
to the PFB. 

Copyright (C) 2014, Square Kilometer Array (SKA) South Africa
@author Benjamin Hugo (bennahugo __AT__ aol __DOT__ com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "inv_pfb.h"
/****************************************************************************************************************
 variables
*****************************************************************************************************************/
cufftReal * d_ifft_output;
float * d_taps;
cufftHandle ifft_plan;
cufftComplex * d_ifft_input;
float * d_filtered_output;

/****************************************************************************************************************
Forward declare kernels
****************************************************************************************************************/
__global__ void ipfb(const cufftReal * input,  float * output, const float * prototype_filter);
__global__ void move_last_P_iffts_to_front(cufftReal * ifft_array, uint32_t start_of_last_P_N_block);

/**
This method initializes the device. It selects the GPU, allocates all the memory needed to perform the inverse pfb process and copies the prototype filter
onto the device. ***WARNING***: this method must be called before the first stride is processed. The device should be released after all processing is completed.
@args taps pointer to a preloaded prototype filter (preferably a hamming windowed FIR filter with cutoff at 1/N)
@postcondition device (if any) is ready to perform the inverse pfb process on multiple strides of data.
*/
void initDevice(const float * taps){
	//do some checks to see if we're initing a reasonable gpu setup:
	assert((LOOP_LENGTH % FFT_SIZE) == 0); //The data being uploaded to the device must consist of an integral number of FFT blocks.

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
		cudaSetDevice(max_device); //select device
        	cudaDeviceReset(); //ensure device is in a safe state before we begin processing
		
		//print some stats:
        	cudaDeviceProp properties;
	        cudaGetDeviceProperties(&properties, max_device);

        	size_t mem_tot = 0;
	        size_t mem_free = 0;
	        cudaMemGetInfo  (&mem_free, & mem_tot);
        	printf("---------------------------------------------------------\n\033[0;31mChosen GPU Statistics\033[0m\n---------------------------------------------------------\n");
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
	  Weighted Window Overlap Add method. TODO: One optimization trick that one may try is to copy this into texture memory
	  where there may be a slight performance increase due to the texture caching properties of the GPU.
	*/
	printf("\033[0;31mInitialization routines\033[0m\n---------------------------------------------------------\nINIT: Copying prototype filter of %d taps to device\n",WINDOW_LENGTH);
	cudaSafeCall(cudaMalloc((void**)&d_taps,sizeof(float)*WINDOW_LENGTH));
	cudaSafeCall(cudaMemcpy(d_taps,taps,sizeof(float)*WINDOW_LENGTH,cudaMemcpyHostToDevice));
	/*
	 Setup the ifft buffer where we will keep P extra IFFTs (each of length N) carried over
	 from the previous stride-processing iteration.
	*/
	printf("INIT: Setting up IFFT output buffer of length %d\n", BUFFER_LENGTH);
	cudaSafeCall(cudaMalloc((void**)&d_ifft_output,sizeof(cufftReal) * (BUFFER_LENGTH)));
	//Setup CU IFFT plan to process all the N-sample iffts contained in a single loop, in one go
	uint32_t max_no_blocks = LOOP_LENGTH/FFT_SIZE;
	printf("INIT: Setting up IFFT plan for %d blocks of FFTs\n",max_no_blocks);
        cufftSafeCall(cufftPlan1d(&ifft_plan, N,  CUFFT_C2R, max_no_blocks));
	//alloc space for the ifft input vector on the device. The input vector should be BATCH * (N/2 + 1) samples long (consisting of complex numbers)
	printf("INIT: Setting up IFFT input vector for %d blocks of FFTs, each with %d non-redundant samples\n",max_no_blocks,FFT_SIZE);
        cudaSafeCall(cudaMalloc((void**)&d_ifft_input,sizeof(cufftComplex) * (FFT_SIZE * max_no_blocks)));
        //reserve memory for output (this should be BATCH * N real samples long)
	printf("INIT: Setting up PFB output vector of to store %d blocks of output, each with %d samples\n",max_no_blocks,N);
        cudaSafeCall(cudaMalloc((void**)&d_filtered_output,sizeof(float) * (max_no_blocks*N)));
	printf("---------------------------------------------------------\n");
}
/**
 Deallocates any memory associated with the inverse pfb process from the device.
 @precondition device should have been initialized before this method is called
*/
void releaseDevice(){
	printf("\033[0;31mAll done, releasing device\033[0m\n---------------------------------------------------------\n");
	cudaSafeCall(cudaFree(d_taps));
	cudaSafeCall(cudaFree(d_ifft_output));
	cudaSafeCall(cudaFree(d_ifft_input));
        cudaSafeCall(cudaFree(d_filtered_output));
	cufftSafeCall(cufftDestroy(ifft_plan));
	cudaDeviceReset(); //leave the device in a safe state
	printf("DEINIT: Device safely released\n---------------------------------------------------------\n");
}

//TODO:REMOVE THIS:
uint32_t writeDataToDiskTEMP(const char * filename, uint32_t element_size, uint32_t length, const void * buffer, bool blankfile){
        FILE * hnd = fopen(filename, blankfile ? "w" : "a");
        uint32_t elemsWrote = 0;
        if (hnd != NULL){
                elemsWrote = fwrite(buffer,element_size,length,hnd);
                fclose(hnd);
        }
        return elemsWrote;
}


/**
This method computes the inverse polyphase filter bank (pfb) operation (Weighted Window Overlap Add method) on a subset ("stride") 
of the output of a previous forward pfb operation.

Background theory:
        For the forward process see:
                https://casper.berkeley.edu/wiki/The_Polyphase_Filter_Bank_Technique
        The inverse process has its subfilters flipped, but follows the same trend, see:
                A REVIEW OF POLYPHASE FILTER BANKS AND THEIR APPLICATION --- Daniel Zhou. 
                Air Force Research Laboratory. AFRL-IF-RS-TR-2006-277 In-House Final 
                Technical Report, September 2006.

The outline of the process is as follows:
 1. Perform N-element IFFTs on all the blocks in this stride. The output has an initial padding of P * N blocks.
 2. Filter the IFFTed samples (N * no_blocks_in_stride) with the inverse filter. This filtering operation starts
 at the beginning of the IFFT_output array and stops P*N samples short of the last index in the IFFT_output array. This
 is due to the fact that the filter looks ahead to compute samples at its current position.
 3. Move the last P*N IFFT samples to the start of the IFFT_output_array to maintain the state of the inverse pfb operation
 for processing the next stride of pfb output.

NOTE: the last point implies that this method has state associated with it. It is critical to maintain a persistant IFFT 
buffer on the cuda device to ensure that we do not loose P*N samples between processing consecutive strides of pfb output.

We could have equivalently achieved step 3 using a ringbuffer. However this would have meant that we could not process
all the IFFTs in one batch (it would have been split up into two batches). Copying P*N samples from one memory location
to another ***ON THE DEVICE*** should be relatively quick (considering we do not have to do a memory copy over a relatively
slow PCI-e bus. This also helps us get rid of the indexing nightmares associated with maintaining a ring buffer.

The initial setup and tairdown costs associated with memory allocation + deallocation is mitigated through initializing the
device once off before processing starts and tairing down the memory allocations only after all processing stops.

@args input list of complex numbers as output by a pfb process (these are interleved IEEE 32-bit floating point numbers, of the
	form 'riririri...'). Note that although this output is produced by an N-point real FFT operation only the first N/2 + 1 samples
	are useful (by the Hermite-symmetric property of the real FFT). The input to this method should therefore be an integral
	number of N/2+1 point FFTs.
@args output_buffer preallocated buffer of size no_blocks_in_stride*N in which the inverse pfb output will be dumped
@args no_blocks_in_stride The integral number of FFT blocks passed to this method. no_blocks_in_stride should be less than or
	equal to LOOP_LENGTH/(N/2+1)
@precondition call initDevice BEFORE calling this method
*/
void processNextStride(const complex_float * input, float * output_buffer, uint32_t no_blocks_in_stride){
	assert(no_blocks_in_stride <= LOOP_LENGTH/(N/2+1)); //this is the maximum number of blocks we can sent to the GPU
	//copy everything in this stride into the device ifft input vector
	printf("Copying %d blocks of FFT data, each of length %d to the device\n", no_blocks_in_stride,FFT_SIZE);
        cudaSafeCall(cudaMemcpy(d_ifft_input,input,sizeof(complex_float) * FFT_SIZE * no_blocks_in_stride,cudaMemcpyHostToDevice));
	printf("Executing batched IFFT on data and saving with offset %d\n",PAD);
	//ifft the data:
	{
		cufftSafeCall(cufftExecC2R(ifft_plan,d_ifft_input,d_ifft_output + PAD));
		
		//TODO: remove this:
		float * ifft_out = (float *)malloc(sizeof(float)*(no_blocks_in_stride * N + PAD));
		
		cudaSafeCall(cudaMemcpy(ifft_out,d_ifft_output, sizeof(cufftReal) * (no_blocks_in_stride * N + PAD),cudaMemcpyDeviceToHost));
		writeDataToDiskTEMP("/home/bhugo/ska-res/ska-ddc/inv_pfb/IFFTedStuff_c.dump",sizeof(float),no_blocks_in_stride * N + PAD,ifft_out,true);
		free(ifft_out);	
	}
	cudaThreadSynchronize();
	printf("Performing inverse filtering, %d threads per block for %d blocks\n",N,no_blocks_in_stride);
	//now do the inverse filtering
	{
		dim3 threadsPerBlock(N,1,1);
		dim3 numBlocks(no_blocks_in_stride,1,1);
		ipfb<<<numBlocks,threadsPerBlock,0>>>(d_ifft_output,d_filtered_output,d_taps);
	
		cudaThreadSynchronize();
		cudaError code = cudaGetLastError();
		if (code != cudaSuccess){
			fprintf (stderr,"Error while executing inverse pfb -- %s\n", cudaGetErrorString(code)); 
			exit(-1);
		}
	}
	printf("Moving %d IFFT samples from position %d of the IFFT persistant buffer to index 0 of the buffer.\n",N * P,N * no_blocks_in_stride);
	//move the last PAD samples in the ifft output array to the front of the ifft output array for processing the next stride of elements:
	{
		dim3 threadsPerBlock(N,1,1);
		dim3 numBlocks(P,1,1);
		move_last_P_iffts_to_front<<<numBlocks,threadsPerBlock,0>>>(d_ifft_output, N * no_blocks_in_stride);
	
		cudaThreadSynchronize(); //Optimization TODO: can do the output memcpy and this kernel asyncly
	        cudaError code = cudaGetLastError();
        	if (code != cudaSuccess){
                	fprintf (stderr,"Error while executing inverse pfb -- %s\n", cudaGetErrorString(code));
                	exit(-1);
		}
	}
	//copy N-sized chunks to the output array:
	printf("Finished inverse pdb on stride, copying %d blocks, each of %d elements from the device\n",no_blocks_in_stride,N);
	cudaMemcpy(output_buffer, d_filtered_output,sizeof(float)*(no_blocks_in_stride*N),cudaMemcpyDeviceToHost);
}

/**
This kernel computes the filter bank operation of the inverse Polyphase Filter Bank (Weighted Window Overlap Add method). 
The kernel should be invoked with blockDim = N and numBlocks = number_of_blocks_in_stride.

It will perform the following filterbank algorithm:
for (l = 0; l < stride_length; l += N) in parallel
	for (n = 0; n < N; ++n) in parallel
		accum = x[l+n]h[N - n - 1]
		for (p = 1; p < P; ++p)
			accum += x[l+n+p*N]h[p*N + (N - n - 1)]
		y[l+n] = accum
		endfor
	endfor
endfor

Technically the prototype filter remains the same for the synthesis filter bank. Each subfilter, however has to be read in reverse. Furthermore
the commutator technically runs in reverse as well, meaning that we should flip the order subfilters are executed in the bank. But whether we accumulate
each y[n] in forward or reverse does not matter. It should be clear that if the 3rd loop is run backwards with the the initialization of the 
accumulation set to a position in the last subfilter the result should remain the same.
*/
__global__ void ipfb(const cufftReal * input,  float * output, const float * prototype_filter){
	uint32_t lB = blockIdx.x * N;
	uint32_t filter_index = N - threadIdx.x - 1;
	register float accum = input[lB + threadIdx.x]*prototype_filter[filter_index]; //Fetching data from both the filter and the input should be coalesced
	#pragma unroll
	for (uint32_t p = 1; p < P; ++p)
		accum += input[lB + threadIdx.x + p*N]*prototype_filter[p*N + filter_index]; //Fetching data from both the filter and the input should be coalesced
	output[lB + threadIdx.x] = accum;
}

/**
This kernel moves the last P*N iffts to the front of the ifft array, so that we can maintain the state of our filterbank between successive strides. It is critical
to call this kernel after the ipfb kernel has fully completed to ensure that the next stride isn't missing P*N samples.

The kernel should be invoked with blockDim = N and numBlocks = P
*/
__global__ void move_last_P_iffts_to_front(cufftReal * ifft_array, uint32_t start_of_last_P_N_block){
	uint32_t tI = blockIdx.x*blockDim.x + threadIdx.x;
	ifft_array[tI] = ifft_array[start_of_last_P_N_block + tI];
} 
