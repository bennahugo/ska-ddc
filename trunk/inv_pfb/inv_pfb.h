#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufft.h>
#include <assert.h>
#include "math.h"
#include "file_reader.h"

#ifndef INV_PFB_H
#define INV_PFB_H

/****************************************************************************************************************
 constants
*****************************************************************************************************************/
const uint16_t N = 512; //Number of FFT samples
const uint16_t P = 8; //Number of Filterbanks
const uint32_t WINDOW_LENGTH = N*P;
const uint32_t PAD = N*P;
const uint32_t LOOP_LENGTH = 1024 * (N/2 + 1); //Size of chunk to send off to the GPU. This number must be divisable by FFT_SIZE (we should send an integral number of FFTs to the GPU).
const uint32_t BUFFER_LENGTH = LOOP_LENGTH + PAD; //Number of elements in the persistant ifft output buffer
const uint32_t FFT_SIZE = N/2 + 1; //size of each input FFT (non-redundant samples)

/****************************************************************************************************************
 debugging flags
*****************************************************************************************************************/


/****************************************************************************************************************
 necessary typedefs
*****************************************************************************************************************/
typedef struct {
        float r;
        float i;
} complex_float;

/****************************************************************************************************************
 CUDA error handling macros
****************************************************************************************************************/
#define cudaSafeCall(value) {                                                                                   \
        cudaError_t _m_cudaStat = value;                                                                                \
        if (_m_cudaStat != cudaSuccess) {                                                                               \
                fprintf(stderr, "Error %s at line %d in file %s\n",                                     \
                                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);           \
                exit(1);                                                                                                                        \
        } }

inline void __cufftSafeCall( uint32_t err, const char *file, const int line ){
        if ( CUFFT_SUCCESS != err ){
                fprintf( stderr, "cufftSafeCall() failed at %s:%i\n", file, line);
                exit( -1 );
        }
        return;
}
#define cufftSafeCall(err)  __cufftSafeCall(err, __FILE__, __LINE__)

/****************************************************************************************************************
 Forward declarations
*****************************************************************************************************************/
void initDevice(const float * taps);
void releaseDevice();
void processNextStride(const complex_float * input, float * output_buffer, uint32_t no_blocks_in_stride = LOOP_LENGTH/(N/2+1));
__global__ void ipfb(const cufftReal * input,  float * output, const float * prototype_filter);
__global__ void move_last_P_iffts_to_front(cufftReal * ifft_array, uint32_t start_of_last_P_N_block);

/****************************************************************************************************************
 variables
*****************************************************************************************************************/
cufftReal * d_ifft_output;
float * d_taps;
cufftHandle ifft_plan;
cufftComplex * d_ifft_input;
float * d_filtered_output;

#endif //ifndef INV_PFB_H
