#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>
#include "cuPrintf.cu"

#define UPSAMPLE  4
#define DOWNSAMPLE 25
#define SAMPLES_PER_THREAD 160
#define LO_FREQ 220e6
#define SAMPLE_FREQ 800e6
#define TEMP_SIZE UPSAMPLE * SAMPLES_PER_THREAD
#define THREADS_PER_BLOCK 1024
#define BLOCKS_PER_GRID 400 * UPSAMPLE * DOWNSAMPLE
 // 256 threads per grid is ideal. 2048 grids per block is a guesstimate
#define BLOCK_SIZE THREADS_PER_BLOCK * BLOCKS_PER_GRID
#define LOOPS 1
#define SIN_TABLE_LENGTH 4096
 // how many loops of block size to do

cudaEvent_t t_start, t_stop;

__constant__ float cTaps[512];
__constant__ float cSin[SIN_TABLE_LENGTH];

__global__ void mix(char *input_buffer, char *output_buffer, int lo_interval, int sin_table_length)
{
 int idx = blockIdx.x*blockDim.x + threadIdx.x;
 output_buffer[idx] = int(((float)cSin[((sin_table_length/4) + lo_interval*idx) % sin_table_length] * (float)input_buffer[idx]));

 // shared approach seems a little slower, perhaps due to occupancy...

 //__shared__ char shared_input[THREADS_PER_BLOCK];
 //shared_input[threadIdx.x] = input_buffer[idx];
 //__syncthreads();
 //output_buffer[idx] = int(((float)cSin[((sin_table_length/4) + lo_interval*idx) % sin_table_length] * (float)shared_input[threadIdx.x]));
}

__global__ void poly_fir(char *input_buffer, float *output_buffer, int no_taps, int upsample, int downsample, int poly_interval, int lo_interval, int sin_table_length)
{
 __shared__ char shared_input[13312];
  // this is sized so that (no_taps / upsample) + (poly_interval * (upsample-1)) + (threadIdx.x * downsample) < 16384 - 10
  // for current example this gives 32 + 18 + (256 * 25) = 6450 < 16374
  // The zero position is memory size from the start (e.g. 50 bytes). This represents the prior input for this filter run
  // and should fix boundary issues.
  // No of output samples from this is X * 4. e.g. 256 * 4 = 1024. input_samples / output_samples = 6.25 = 25 / 4 = downsample / upsample
  // For expedience we get each thread to contribute a single float to the front of the buffer to make the memory_size segment.
  // This is redundant by blockDim.x - memory_size. So our size becomes 256 * 26 = 6656
 int memory_size = blockDim.x; //(no_taps / upsample) + (poly_interval * (upsample-1));

 if (blockIdx.x > 0) shared_input[threadIdx.x] = input_buffer[blockIdx.x*blockDim.x*downsample + threadIdx.x - blockDim.x];
 else shared_input[threadIdx.x] = 0;
 int idx = blockIdx.x*blockDim.x + threadIdx.x;

 for (int k=0; k < downsample; k++) {
  shared_input[memory_size+threadIdx.x*downsample+k] = input_buffer[blockIdx.x*blockDim.x*downsample + threadIdx.x*downsample + k];
  //shared_input[memory_size+threadIdx.x*downsample+k] = int(((float)cSin[((sin_table_length/4) + lo_interval*idx) % sin_table_length] * (float)input_buffer[blockIdx.x*blockDim.x*downsample + threadIdx.x*downsample + k]));
 }

 __syncthreads();
 float temp_output = 0;
 int fir_idx = (threadIdx.x*downsample + memory_size) + ((threadIdx.y) * poly_interval);
  // each x thread steps in units of 25 samples. The y thread steps through the four fir filters. We also add the memory_size to skip the first block.
#pragma unroll 32
 for (int i=threadIdx.y; i < no_taps; i+=blockDim.y) {
  temp_output += cTaps[i] * (float)shared_input[fir_idx - (i/blockDim.y)];
 }
 output_buffer[blockIdx.x*blockDim.x*blockDim.y + threadIdx.x*blockDim.y + threadIdx.y] = temp_output;
  // output index is chunked by numnber of y threads
}

__global__ void mix_up(char *input_buffer, float *sin_table, float *output_buffer, int upsample, int lo_interval, int sin_table_length)
{
 // producs an mixed block that has length 4*len(input_buffer), padded with zeros
 int idx = blockIdx.x*blockDim.x + threadIdx.x;
 //if (threadIdx.x == 0) { 
  cuPrintf("blockIdx.x: %d, threadIdx.x: %d\n", blockIdx.x, threadIdx.x);
 //}
 output_buffer[blockIdx.x*blockDim.x*upsample + threadIdx.x*upsample] = sin_table[(lo_interval*idx) % sin_table_length] * (int)input_buffer[idx];
}

__global__ void fir_shared(float *input_buffer, float *output_buffer, int no_taps, int upsample, int downsample)
{
 __shared__ float shared_input[THREADS_PER_BLOCK*2];
  shared_input[threadIdx.x] = input_buffer[blockIdx.x*blockDim.x + threadIdx.x - THREADS_PER_BLOCK];
   // this is pretty inefficient as we only need no_taps worth of prepend data as opposed to THREADS_PER_BLOCK
   // input buffer also needs to be zero padded by THREADS_PER_BLOCK
  __syncthreads();
  shared_input[threadIdx.x + THREADS_PER_BLOCK] = input_buffer[blockIdx.x*blockDim.x + threadIdx.x];
  __syncthreads();

 int fir_idx = 0;
 float temp_output = 0;

#pragma unroll 128
 for (int i=0; i < no_taps; i++) {
  fir_idx = THREADS_PER_BLOCK + threadIdx.x - i;
  temp_output += cTaps[i] * shared_input[fir_idx];
  //temp_output += cTaps[i] * input_buffer[blockIdx.x*blockDim.x + threadIdx.x - i];
 }
 __syncthreads();
 output_buffer[blockIdx.x*blockDim.x + threadIdx.x] = temp_output / 127;
}

__global__ void down(float *input_buffer, float *output_buffer, int downsample)
{
 int out_idx = blockIdx.x*blockDim.x + threadIdx.x;
 output_buffer[out_idx] = input_buffer[out_idx * downsample];
}

__global__ void float_cast(float *in, char *out)
{
 int idx = blockDim.x*blockIdx.x + threadIdx.x;
 out[idx] = (char) in[idx];
}

int main(int argc, char **argv) {
 int write_block = BLOCK_SIZE;
  // 1 MB worth of data at a time...
 int samples_per_thread = SAMPLES_PER_THREAD;
  // how many input data points each thread will consume.
  // this should be enough to allow at least a couple of output points per thread when tap overlap 
 int sample_rate = SAMPLE_FREQ;
  // our adc sampling frequency
 int lo_freq = LO_FREQ;
  // the mixing frequency for the DDC
 int upsample = UPSAMPLE;
 int downsample = DOWNSAMPLE;
  // coefficients to sort out the output sample rate
  // in this case giving us 128 MHz sampling
 int sin_table_length = SIN_TABLE_LENGTH;
  // the number of samples in the sin lookup table
 int sin_table_size = sizeof(float) * sin_table_length;
 int lo_interval = int(((float)sin_table_length / sample_rate) * lo_freq);
  // the stepping interval through the lo sin table. May result in slightly different lo_freq from that
  // specified. The user is informed of this.
 int lo_offset = 0;
  // as we move from block to block our we need an lo_offset to maintain phase...
 int no_taps = 0;
  // number of filter taps. Calculated once filter data is loaded.
 int no_output_samples = int(((float)upsample / downsample) * write_block);
  // overall number of output samples to produce for this block

 int fh;
 char *data_file;
 char *fir_taps_file;
 short first = 1;
 long int start = 0;
 float et;
 struct stat stat_buf;

 float *fir_taps;
 char *base_buffer;
 char *host_char_buffer;
 float *output_buffer;
 float *upsample_buffer;
 float *sin_table;
  // host buffers

 char *device_char_buffer;
 float *device_fir_taps;
 float *device_output_buffer;
 float *device_upsample_buffer;
 float *device_float_buffer;
 float *device_fir_buffer;
 float *device_sin_table;
  // device buffers

 if (argc > 2) {
  data_file = argv[1];
  fir_taps_file = argv[2];
 } else { printf("Please supply both data and fir_taps filenames...\n"); return -1;}

 fprintf(stderr,"Actual lo freq is: %f MHz (%i, %i, %i, %i)\n", lo_interval / ((float)sin_table_length / sample_rate), lo_interval, sin_table_length, sample_rate, lo_freq);
 fprintf(stderr,"Producing %i output samples per block (%i samples).\n",no_output_samples,write_block);

 base_buffer = (char*)malloc(write_block*LOOPS);
 host_char_buffer = (char*)malloc(no_output_samples);
 output_buffer = (float*)malloc(sizeof(float) * no_output_samples);
 upsample_buffer = (float*)malloc(sizeof(float) * upsample * (write_block + THREADS_PER_BLOCK));
 sin_table = (float*)malloc(sin_table_size);
 memset(host_char_buffer, (char) 0, no_output_samples);
 memset(upsample_buffer, (char) 0, sizeof(float) * upsample * (write_block + THREADS_PER_BLOCK));
  // zero as we use part of this for our initial zero padding block

 fh = open(fir_taps_file, O_RDONLY);
 fstat(fh, &stat_buf);
 no_taps = stat_buf.st_size / sizeof(float);
 fprintf(stderr,"Using %i tap FIR filter.\n",no_taps);
 fir_taps = (float*)malloc(sizeof(float) * no_taps);
 read(fh, fir_taps, sizeof(float) * no_taps);
 close(fh);

 fprintf(stderr,"Preparing sin lookup table...\n");
 for (int i=0; i < sin_table_length; i++) {
  sin_table[i] = sin(i * (2*M_PI/sin_table_length));
 }

 fprintf(stderr,"Allocating block storage on GPU...\n");
 cudaPrintfInit();

 cudaEventCreate(&t_start);
 cudaEventCreate(&t_stop);

 cudaMalloc((void**)&device_char_buffer, write_block);
  // device buffer with space for initial zero padding
 cudaMalloc((void**)&device_output_buffer, sizeof(float) * no_output_samples);
 cudaMalloc((void**)&device_float_buffer, sizeof(float) * write_block);
 cudaMalloc((void**)&device_upsample_buffer, sizeof(float) * upsample * (write_block + THREADS_PER_BLOCK));
 cudaMalloc((void**)&device_fir_buffer, sizeof(float) * upsample * write_block);
 cudaMalloc((void**)&device_sin_table, sin_table_size);
 cudaMalloc((void**)&device_fir_taps, sizeof(float) * no_taps);
  // allocate the device storage

 fprintf(stderr,"Copying FIR taps and SIN lookup table to GPU...\n");
 cudaMemcpy(device_fir_taps, fir_taps, sizeof(float) * no_taps, cudaMemcpyHostToDevice);
  // copy the filter taps to the device
 cudaMemcpyToSymbol(cTaps, fir_taps, sizeof(float) * no_taps);
 cudaMemcpyToSymbol(cSin, sin_table, sin_table_size);
 cudaMemcpy(device_sin_table, sin_table, sin_table_size, cudaMemcpyHostToDevice);
 cudaMemcpy(device_upsample_buffer, upsample_buffer, sizeof(float) * upsample * write_block, cudaMemcpyHostToDevice);
 cudaMemcpy(device_output_buffer, upsample_buffer, sizeof(float) * no_output_samples, cudaMemcpyHostToDevice);
 cudaMemcpy(device_fir_buffer, upsample_buffer, sizeof(float) * upsample * write_block, cudaMemcpyHostToDevice);
 cudaMemcpy(device_float_buffer, upsample_buffer, sizeof(float) * write_block, cudaMemcpyHostToDevice);

 fprintf(stderr,"GPU Configuration: blocks per grid: %i, threads per block: %i\n",BLOCKS_PER_GRID, THREADS_PER_BLOCK);

 fh = open(data_file, O_LARGEFILE);
 read(fh, base_buffer, write_block * LOOPS);
 fprintf(stderr,"Reading %i bytes of data...\n", write_block * LOOPS);

 for (int i=0; i < LOOPS; i++) {

  start = i * write_block;
  fprintf(stderr,"Loop %i (start: %li).\n",i,start);

  cudaEventRecord(t_start, 0);

  cudaMemcpy(device_char_buffer, base_buffer+start, write_block, cudaMemcpyHostToDevice);
   // need to recal lo_offset each loop
  lo_offset = 0;

   // polyphase method
  int poly_interval = 6;
   // hardcoded for now... l = 4; m = 25
  dim3 threads(THREADS_PER_BLOCK / upsample, upsample);
   // the downsample spaced blocks are indexed by thread.x and the upsample number of fir filters are indexed by y
  dim3 blocks(BLOCKS_PER_GRID / downsample,1);

  mix<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(device_char_buffer, device_char_buffer, lo_interval, sin_table_length);
  poly_fir<<<blocks, threads>>>(device_char_buffer, device_output_buffer, no_taps, upsample, downsample, poly_interval, lo_interval, sin_table_length);

   // brute force
  //mix_up<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(device_char_buffer, device_sin_table, device_upsample_buffer, upsample, lo_interval, sin_table_length);
  //fir_shared<<<BLOCKS_PER_GRID * upsample,THREADS_PER_BLOCK>>>(device_upsample_buffer, device_fir_buffer, no_taps, upsample, downsample);
  //down<<<int(BLOCKS_PER_GRID*upsample/downsample),THREADS_PER_BLOCK>>>(device_fir_buffer, device_output_buffer, downsample);

  float_cast<<<no_output_samples/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(device_output_buffer, device_char_buffer);
  cudaMemcpy(host_char_buffer, device_char_buffer, no_output_samples, cudaMemcpyDeviceToHost);
  //cudaMemcpy(output_buffer, device_output_buffer, sizeof(float) * no_output_samples, cudaMemcpyDeviceToHost);

  cudaEventRecord(t_stop, 0);
  cudaEventSynchronize(t_stop);
  cudaEventElapsedTime(&et, t_start, t_stop);
  for (int j=0; j < 10; j++) { fprintf(stderr,"Expanded: Char buffer value %i is %i\n",j, host_char_buffer[j]); }
  write(1, host_char_buffer, no_output_samples);
  //write(1, output_buffer, sizeof(float) * no_output_samples);
  fprintf(stderr,"Loop done. CUDA time is %f ms\n", et);
 }
 read(1, output_buffer, 1);
 return 0;
}
