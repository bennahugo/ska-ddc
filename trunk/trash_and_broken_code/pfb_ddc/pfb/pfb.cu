#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cufft.h>
#include <math.h>

#define BLOCK_SIZE 1024*1024
#define LOOPS 10
 // how many loops of block size to do

cudaEvent_t t_start, t_stop;
cufftHandle plan;

__global__ void filter_block(char *input_buffer, float *taps, float *output_buffer, int N, int P)
{
 float temp_output = 0;

 for (int x=0; x < P; x++) {
  temp_output += taps[threadIdx.x + x*N]*(int)input_buffer[(P*N) + threadIdx.x - x*N + (blockIdx.x*blockDim.x)];
   // input buffer of continuous voltage samples should be distributed amongst N channels
   // index into input buffer:
   //     current thread index indicates which channel we are currently processing
   //     x which tap we are getting data for
   //     blockIdx.x * blockDim.x is our depth into the block of data in multiples of channel number
   // structure of taps vector is actually [N,P]. So each thread will read taps[channel_number:channel_number+8] for taps
 }
 output_buffer[threadIdx.x + (blockIdx.x*blockDim.x)] = temp_output;
}

 __global__ void float_cast(float *in, char *out, int divider)
{
 int idx = (blockDim.x * divider) * blockIdx.x + threadIdx.x;
 out[blockDim.x*blockIdx.x + threadIdx.x] = (char) (in[idx + (2 * blockIdx.x)] / 512);
  // skip the (N/2 +1)'th fft output...
}

int main(int argc, char **argv) {
 int write_block = BLOCK_SIZE;
  // 1 MB worth of data at a time...
 int N = 512;
  // # of frequency channels in the PFB
 int P = 8;
  // length of pre filtering
 int divider = 1;
  // output divider to reduce rate
 int write_size = sizeof(float) * write_block;
 int tap_size = sizeof(float) * P * N;
 int fft_output_size = (write_block / N) * sizeof(cufftComplex) * (N/2 + 1);
  // hold BATCH (write_block / N) worth of complex FFT outputs
 int fh;
 unsigned int i=0;
 char *data_file;
 char *fir_taps_file;
 char *base_buffer;
 float et;
  // counter for elapsed time of cuda ops
 float *fir_taps;
 char *device_char_buffer;
 char *host_char_buffer;
 float *device_float_buffer;
 cufftComplex *device_complex_buffer;
 float *device_fir_taps;
 short first = 1;
 long int start = 0;

 if (argc > 2) {
  data_file = argv[1];
  fir_taps_file = argv[2];
 } else { printf("Please supply both data and fir_taps filenames...\n"); return -1;}

 fprintf(stderr,"Sizes %li, %li\n", sizeof(char), sizeof(int));
 base_buffer = (char*)malloc(write_block*LOOPS);
 host_char_buffer = (char*)malloc(write_block);
 memset(host_char_buffer, (char) 0, write_block);
  // zero as we use part of this for our initial zero padding block

 float *float_buffer = (float*)malloc(fft_output_size);

 fir_taps = (float*)malloc(tap_size);

 fh = open(fir_taps_file, O_RDONLY);
 read(fh, fir_taps, tap_size);
  // source of taps vector should be flattened [P,N] array
 close(fh);
 //for (i=0; i < P*N; i++) { fprintf(stderr,"%f ",(float)*(fir_taps+i));}

 cudaEventCreate(&t_start);
 cudaEventCreate(&t_stop);

 cudaMalloc((void**)&device_char_buffer, write_block + (P*N));
 cudaMalloc((void**)&device_float_buffer, write_size);
 cudaMalloc((void**)&device_complex_buffer, fft_output_size);
 cudaMalloc((void**)&device_fir_taps, tap_size);
  // allocate the device storage

 cudaMemcpy(device_fir_taps, fir_taps, tap_size, cudaMemcpyHostToDevice);
  // copy the filter taps to the device

 int threadsPerBlock = N;
 int blocksPerGrid = write_block / N;
 fprintf(stderr,"Blocks per grid: %i, Threads per block: %i\n",blocksPerGrid, threadsPerBlock);
 cufftPlan1d(&plan, N, CUFFT_R2C, int(write_block/N));
 fprintf(stderr,"FFT Plan has length %i with batch size %i\n",N, int(write_block/N));

 fh = open(data_file, O_LARGEFILE);
 read(fh, base_buffer, write_block * LOOPS);
 fprintf(stderr,"Reading %i bytes of data...\n", write_block * LOOPS);

 for (i=0; i < LOOPS; i++) {

  start = i * write_block;
  fprintf(stderr,"Loop %i (start: %li).\n",i,start);

  cudaEventRecord(t_start, 0);

  if (first == 1) {
   fprintf(stderr,"Appending zero pad block to start of stream...\n");
   cudaMemcpy(device_char_buffer, host_char_buffer, P*N, cudaMemcpyHostToDevice);
    // copy the pad block in first (we have zero'ed this block earlier)
   cudaMemcpy(device_char_buffer+(P*N), base_buffer + start, write_block, cudaMemcpyHostToDevice);
   first = 0;
  } else {
   cudaMemcpy(device_char_buffer, base_buffer+start - (P*N), write_block + (P*N), cudaMemcpyHostToDevice);
    // copy data to GPU. Add extra N*P block for boundary handling...
  }

  filter_block<<<blocksPerGrid, threadsPerBlock>>>(device_char_buffer, device_fir_taps, device_float_buffer, N, P);
   // kernel applies pre filtering to entire block leaving it in device_output_buffer ready for FFT
  cufftExecR2C(plan, (cufftReal*)device_float_buffer, (cufftComplex*)device_complex_buffer);
   // Do FFT's over the entire block, one column at a time
  //cudaMemcpy(float_buffer, device_complex_buffer, fft_output_size, cudaMemcpyDeviceToHost);
   // get the intermediate results...
  //write(1, float_buffer, fft_output_size);
   // output the intermediate results...
  float_cast<<<write_block/N, N/divider>>>((float*)device_complex_buffer, device_char_buffer, divider);
   // prepare our output stream...
  cudaMemcpy(host_char_buffer, device_char_buffer, write_block/divider, cudaMemcpyDeviceToHost);
   //for (i=0; i < 100 * (N/2 + 1); i++) { fprintf(stderr,"Complex value %i has x=%f, y=%f\n", i, fft_buffer[i].x, fft_buffer[i].y); }
  cudaEventRecord(t_stop, 0);
  cudaEventSynchronize(t_stop);
  cudaEventElapsedTime(&et, t_start, t_stop);
  fprintf(stderr,"Loop done. CUDA time is %f ms\n", et);
  write(1, host_char_buffer, write_block/divider);
   // emit to stdout (which has hopefully been redirected...)
 }
 return 0;
}
