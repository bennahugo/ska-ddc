#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>

#define DOWNSAMPLE 25
#define SAMPLE_FREQ 800e6
#define LO_FREQ 302e6
#define THREADS_PER_BLOCK 512
#define BLOCKS_PER_GRID 32 * DOWNSAMPLE
 // keeps things a multiple of COS_TABLE_LENGTH to avoid edge effects
#define BLOCK_SIZE THREADS_PER_BLOCK * BLOCKS_PER_GRID
#define SECONDS_OF_DATA 10
#define COS_TABLE_LENGTH 8000
 // how many loops of block size to do
#define INPUT_SCALE 1.0f
 // input may need scaling to see tones clearly...

__constant__ float cTaps[512];
cudaEvent_t t_start, t_stop;
__constant__ float cCos[COS_TABLE_LENGTH];

__global__ void mix(char *input_buffer, float *output_buffer, int lo_interval, int cos_table_length, int lo_offset)
{
 int idx = blockIdx.x*blockDim.x + threadIdx.x;
 float mix_value = (float)cCos[(lo_interval*idx) % cos_table_length + lo_offset] * ((float)input_buffer[idx + blockDim.x] / INPUT_SCALE);

 output_buffer[idx + blockDim.x] = mix_value;
  // when mixing we skip the first blockDim.x (the number of threads) worth of input buffer (i.e. the memory portion)
}

__global__ void simple_fir(float *input_buffer, float *output_buffer, int no_taps, int upsample, int downsample, int poly_interval, int lo_interval, int cos_table_length)
{
 float temp_output = 0;
 int fir_idx = (blockIdx.x*blockDim.x + threadIdx.x)*downsample + blockDim.x;
    // idx starts from end of memory buffer which is blockDim.x in length

 for (int i=0; i < no_taps; i++) {
  temp_output += cTaps[i] * (float)input_buffer[fir_idx - i];
 }
 output_buffer[blockIdx.x*blockDim.x + threadIdx.x] = temp_output * (downsample / 8.0f);
}

__global__ void float_cast(float *in, char *out)
{
 int idx = blockDim.x*blockIdx.x + threadIdx.x;
 out[idx] = (char) in[idx];
}

int main(int argc, char **argv) {
 int write_block = BLOCK_SIZE;
  // 1 MB worth of data at a time...
  // this should be enough to allow at least a couple of output points per thread when tap overlap 
 int sample_rate = SAMPLE_FREQ;
  // our adc sampling frequency
 int upsample = 1;
 int downsample = DOWNSAMPLE;
  // coefficients to sort out the output sample rate
  // in this case giving us 128 MHz sampling
 int lo_freq = LO_FREQ;
  // the mixing frequency for the DDC
 int cos_table_length = COS_TABLE_LENGTH;
  // the number of samples in the sin lookup table
 int cos_table_size = sizeof(float) * cos_table_length;
 int lo_interval = int(((float)cos_table_length / sample_rate) * lo_freq);
  // the stepping interval through the lo sin table. May result in slightly different lo_freq from that
  // specified. The user is informed of this.
 int lo_offset = 0;
  // as we move from block to block our we need an lo_offset to maintain phase...
 int lo_remainder = (write_block * lo_interval) % cos_table_length;
  // the remainder at the end of each loop (i.e. the last point per block from the loop)
 int no_taps = 0;
  // number of filter taps. Calculated once filter data is loaded.
 int no_output_samples = int(((float)write_block / downsample));
  // overall number of output samples to produce for this block
 int loops = int(SECONDS_OF_DATA * (sample_rate / (float)write_block));
 fprintf(stderr,"%i, %i, %i, %i\n", loops, SECONDS_OF_DATA, sample_rate, write_block);

 int fh;
 char *data_file;
 char *fir_taps_file;
 float et;
 struct stat stat_buf;

 float *fir_taps;
 char *base_buffer;
 char *host_char_buffer;
 float *output_buffer;
 float *upsample_buffer;
 float *cos_table;
  // host buffers

 char *device_char_buffer;
 char *memory_buffer;
 float *device_fir_taps;
 float *device_output_buffer;
 float *device_upsample_buffer;
 float *device_float_buffer;
 float *device_fir_buffer;
  // device buffers

 if (argc > 2) {
  data_file = argv[1];
  fir_taps_file = argv[2];
 } else { printf("Please supply both data and fir_taps filenames...\n"); return -1;}

 fprintf(stderr,"Producing %i output samples per block (%i samples).\n",no_output_samples,write_block);

 base_buffer = (char*)malloc(write_block);
 host_char_buffer = (char*)malloc(no_output_samples);
 output_buffer = (float*)malloc(sizeof(float) * no_output_samples);
 upsample_buffer = (float*)malloc(sizeof(float) * upsample * (write_block + THREADS_PER_BLOCK));
 cos_table = (float*)malloc(cos_table_size);
 memset(host_char_buffer, (char) 0, no_output_samples);
 memset(base_buffer, (char) 0, write_block);
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
 for (int i=0; i < cos_table_length; i++) {
  cos_table[i] = 2 * cos(i * (2*M_PI/cos_table_length));
 }

 fprintf(stderr,"Allocating block storage on GPU...\n");

 cudaEventCreate(&t_start);
 cudaEventCreate(&t_stop);

 cudaMalloc((void**)&device_char_buffer, write_block + THREADS_PER_BLOCK);
  // device buffer with space for initial zero padding
 cudaMalloc((void**)&memory_buffer, THREADS_PER_BLOCK);
  // previous loop memory
 cudaMalloc((void**)&device_output_buffer, sizeof(float) * no_output_samples);
 cudaMalloc((void**)&device_float_buffer, sizeof(float) * write_block);
 cudaMalloc((void**)&device_upsample_buffer, sizeof(float) * upsample * (write_block + THREADS_PER_BLOCK));
 cudaMalloc((void**)&device_fir_buffer, sizeof(float) * upsample * write_block);
 cudaMalloc((void**)&device_fir_taps, sizeof(float) * no_taps);
  // allocate the device storage

 cudaMemcpy(device_fir_taps, fir_taps, sizeof(float) * no_taps, cudaMemcpyHostToDevice);
  // copy the filter taps to the device
 cudaMemcpyToSymbol(cTaps, fir_taps, sizeof(float) * no_taps);
 cudaMemcpyToSymbol(cCos, cos_table, cos_table_size);
 cudaMemcpy(device_upsample_buffer, upsample_buffer, sizeof(float) * upsample * write_block, cudaMemcpyHostToDevice);
 cudaMemcpy(device_output_buffer, upsample_buffer, sizeof(float) * no_output_samples, cudaMemcpyHostToDevice);
 cudaMemcpy(device_fir_buffer, upsample_buffer, sizeof(float) * upsample * write_block, cudaMemcpyHostToDevice);
 cudaMemcpy(device_float_buffer, upsample_buffer, sizeof(float) * write_block, cudaMemcpyHostToDevice);
 cudaMemcpy(device_char_buffer, base_buffer, write_block + THREADS_PER_BLOCK, cudaMemcpyHostToDevice);
 cudaMemcpy(memory_buffer, base_buffer, THREADS_PER_BLOCK, cudaMemcpyHostToDevice);
  // init host memory to zero

 fprintf(stderr,"Actual lo freq is: %f MHz (interval: %i, reminader: %i)\n", (lo_interval / ((float)cos_table_length / sample_rate)) / 1e6, lo_interval, lo_remainder);
 fprintf(stderr,"GPU Configuration: blocks per grid: %i, threads per block: %i\n",BLOCKS_PER_GRID, THREADS_PER_BLOCK);

 fh = open(data_file, O_LARGEFILE);
// read(fh, base_buffer, write_block * LOOPS);
 fprintf(stderr,"Producing %.2f s of data (%i loops reading a total of %.2f Mbytes of data)\n", loops * (write_block / float(sample_rate)), loops, loops * (write_block / float(1024*1024)));

 for (int i=0; i <= loops; i++) {
  read(fh, base_buffer, write_block);

  //start = i * write_block;
  //fprintf(stderr,"Loop %i (start: %li, lo_offset: %i).\n",i,start, lo_offset);

  cudaEventRecord(t_start, 0);

  cudaMemcpy(device_char_buffer+THREADS_PER_BLOCK, base_buffer, write_block, cudaMemcpyHostToDevice);
   // need to recal lo_offset each loop
  lo_offset = (i * lo_remainder) % cos_table_length;
   // on the offchance that each block does not loop through the cos table exactly
   // we adjust the starting offset to match the end of the previous loop

   // polyphase method
  int poly_interval = 0;
   // hardcoded for now... l = 4; m = 25
  dim3 threads(THREADS_PER_BLOCK, 1);
   // the downsample spaced blocks are indexed by thread.x and the upsample number of fir filters are indexed by y
  dim3 blocks(BLOCKS_PER_GRID / downsample,1);

  mix<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(device_char_buffer, device_float_buffer, lo_interval, cos_table_length, lo_offset);
  //poly_fir<<<blocks, threads>>>(device_char_buffer, memory_buffer, device_output_buffer, no_taps, upsample, downsample, poly_interval, lo_interval, cos_table_length);
  simple_fir<<<blocks, threads>>>(device_float_buffer, device_output_buffer, no_taps, upsample, downsample, poly_interval, lo_interval, cos_table_length);
  float_cast<<<no_output_samples/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(device_output_buffer, device_char_buffer);
  //fill_memory_buffer<<<1, THREADS_PER_BLOCK>>>(device_char_buffer, memory_buffer, write_block - THREADS_PER_BLOCK);
   // fill memory buffer for next loop
  cudaMemcpy(host_char_buffer, device_char_buffer, no_output_samples, cudaMemcpyDeviceToHost);
  cudaMemcpy(device_char_buffer, base_buffer + write_block - THREADS_PER_BLOCK, THREADS_PER_BLOCK, cudaMemcpyHostToDevice);
   // prime memory buffer for next trip

  cudaEventRecord(t_stop, 0);
  cudaEventSynchronize(t_stop);
  cudaEventElapsedTime(&et, t_start, t_stop);
  if (i == 0) {for (int j=0; j < 20; j++) { fprintf(stderr,"%i ",host_char_buffer[j]); }}
  write(1, host_char_buffer, no_output_samples);
  //write(1, output_buffer, sizeof(float) * no_output_samples);
  //if (i % 20 == 0) fprintf(stderr,"Loop done. CUDA time is %f ms\n", et);
 }
 read(1, output_buffer, 1);
 return 0;
}
