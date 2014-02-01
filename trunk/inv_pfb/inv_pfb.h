#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufft.h>
#include <assert.h>
#include "math.h"

#ifndef INV_PFB_H
#define INV_PFB_H

/****************************************************************************************************************
 constants
*****************************************************************************************************************/
const uint16_t N = 512; //Number of FFT samples (safe to tweak)
const uint16_t P = 8; //Number of Filterbanks (safe to tweak)
const uint32_t FFT_SIZE = N/2 + 1; //size of each input FFT (non-redundant samples)
const uint32_t WINDOW_LENGTH = N*P;
const uint32_t PAD = N*P;
/*Size of chunk to send off to the GPU. Safe to tweak, **BUT**: this number must be divisable by FFT_SIZE (we should 
send an integral number of FFTs to the GPU):
*/
const uint32_t LOOP_LENGTH = 101 * FFT_SIZE;
const uint32_t BUFFER_LENGTH = LOOP_LENGTH / FFT_SIZE * N + PAD; //Number of elements in the persistant ifft output buffer
const uint32_t MAX_NO_BLOCKS = LOOP_LENGTH / FFT_SIZE;

/*Block size of the int8 to cufftReal casting kernel. 256 threads per block seem to be a magic number in CUDA that works 
well accross different generations of cards:
*/
const uint16_t CASTING_THREADS_PER_BLOCK = 256;

/****************************************************************************************************************
 necessary Abstract Data Types
*****************************************************************************************************************/
typedef struct {
        int8_t r;
        int8_t i;
} complex_int8;

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
void processNextStride(const complex_int8 * input, int8_t * output_buffer, uint32_t no_blocks_in_stride = LOOP_LENGTH/(N/2+1));
#endif //ifndef INV_PFB_H
