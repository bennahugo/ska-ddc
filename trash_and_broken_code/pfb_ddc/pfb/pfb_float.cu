#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cufft.h>
#include <math.h>
cudaEvent_t t_start, t_stop;
cufftHandle plan;

__global__ void filter_block(float *input_buffer, float *taps, float *output_buffer, int N, int P)
{
 float temp_output = 0;

 for (int x=0; x < P; x++) {
  temp_output += taps[threadIdx.x + x*N]*input_buffer[threadIdx.x + x*N + (blockIdx.x*blockDim.x)];
   // input buffer of continuous voltage samples should be distributed amongst N channels
   // index into input buffer:
   //     current thread index indicates which channel we are currently processing
   //     x which tap we are getting data for
   //     blockIdx.x * blockDim.x is our depth into the block of data in multiples of channel number
   // structure of taps vector is actually [N,P]. So each thread will read taps[channel_number:channel_number+8] for taps
 } 

 output_buffer[threadIdx.x + (blockIdx.x*blockDim.x)] = temp_output;
}

int main(int argc, char **argv) {
 int write_block = 1024*1024;
  // 1 MB worth of data at a time...
 int N = 512;
  // # of frequency channels in the PFB
 int P = 8;
  // length of pre filtering
 int write_size = sizeof(float) * write_block;
 int tap_size = sizeof(float) * P * N;
 int fft_output_size = (write_block / N) * sizeof(cufftComplex) * (N/2 + 1);
  // hold BATCH (write_block / N) worth of complex FFT outputs
 int fh;
 unsigned int i=0;
 char *data_file;
 char *fir_taps_file;
 char *base_buffer;
 float *float_buffer;
 float et;
  // counter for elapsed time of cuda ops
 float *fir_taps;
 float *device_input_buffer;
 float *device_output_buffer;
 cufftComplex *fft_output_buffer;
 cufftComplex *fft_buffer;
 float *device_fir_taps;

 if (argc > 2) {
  data_file = argv[1];
  fir_taps_file = argv[2];
 } else { printf("Please supply both data and fir_taps filenames...\n"); return -1;}

 base_buffer = (char*)malloc(write_block);
 float_buffer = (float*)malloc(write_size);
 fir_taps = (float*)malloc(tap_size);
 fft_buffer = (cufftComplex*)malloc(fft_output_size);

 fh = open(fir_taps_file, O_RDONLY);
 read(fh, fir_taps, tap_size);
  // source of taps vector should be flattened [P,N] array
 close(fh);
 //for (i=0; i < P*N; i++) { fprintf(stderr,"%f ",(float)*(fir_taps+i));}

 fh = open(data_file, O_LARGEFILE);
 read(fh, base_buffer, write_block);
  // read in a write block worth of int8

 cudaEventCreate(&t_start);
 cudaEventCreate(&t_stop);

 cudaMalloc((void**)&device_input_buffer, write_size);
 cudaMalloc((void**)&device_output_buffer, write_size);
 cudaMalloc((void**)&fft_output_buffer, fft_output_size);
 cudaMalloc((void**)&device_fir_taps, tap_size);
  // allocate the device storage

 cudaMemcpy(device_fir_taps, fir_taps, tap_size, cudaMemcpyHostToDevice);                         
  // copy the filter taps to the device

 int threadsPerBlock = N;
 int blocksPerGrid = write_block / N;
 fprintf(stderr,"Blocks per grid: %i, Threads per block: %i\n",blocksPerGrid, threadsPerBlock);
 cufftPlan1d(&plan, N, CUFFT_R2C, int(write_block/N));
 fprintf(stderr,"FFT Plan has length %i with batch size %i\n",N, int(write_block/N));

 for (i = 0; i < write_block; ++i) {
  *(float_buffer+i) = (float)*(base_buffer+i);
 }

 //write(1, float_buffer, write_size);
 // output the raw data for comparison..
 //for (i=0; i < write_block; i++) { fprintf(stderr,"Base: %i, Float: %f\n",(int)*(base_buffer+i),(float)*(float_buffer+i)); }

 cudaEventRecord(t_start, 0);
 cudaMemcpy(device_input_buffer, float_buffer, write_size, cudaMemcpyHostToDevice);       
  // copy the floats to the device
 filter_block<<<blocksPerGrid, threadsPerBlock>>>(device_input_buffer, device_fir_taps, device_output_buffer, N, P);
  // kernel applies pre filtering to entire block leaving it in device_output_buffer ready for FFT
 cudaMemcpy(float_buffer, device_output_buffer, write_size, cudaMemcpyDeviceToHost);
  // get the intermediate results...
 write(1, float_buffer, write_size);
  // output the intermediate results...
 cufftExecR2C(plan, (cufftReal*)device_output_buffer, (cufftComplex*)fft_output_buffer);
  // Do FFT's over the entire block, one column at a time
 cudaMemcpy(fft_buffer, fft_output_buffer, fft_output_size, cudaMemcpyDeviceToHost);
  // get the final block
 //for (i=0; i < 100 * (N/2 + 1); i++) { fprintf(stderr,"Complex value %i has x=%f, y=%f\n", i, fft_buffer[i].x, fft_buffer[i].y); }
 cudaEventRecord(t_stop, 0);
 cudaEventSynchronize(t_stop);
 cudaEventElapsedTime(&et, t_start, t_stop);
 
 fprintf(stderr,"Done. CUDA time is %f ms\n", et);
 write(1, fft_buffer, write_size);
  // emit to stdout (which has hopefully been redirected...)
 return 0;
}
