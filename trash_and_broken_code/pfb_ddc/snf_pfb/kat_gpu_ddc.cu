/*
 * SPEAD payload capture using Myricom Sniffer API.
 *
 * Also does GPU Digital Down Conversion before writing to disk.
 * 
 * Portions from Myricom Sniffer example code.
 */

#include <inttypes.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <signal.h>
#include <assert.h>
#include <sys/time.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "snf.h"


#ifndef PRIu64
 #define PRIu64 "lu"
#endif

#define UDP_HDRLEN 42
 // ethernet + ip + udp header size
#define PAYLOAD_SIZE 4096       // hardcoded for now
#define DOWNSAMPLE 25
#define SAMPLE_FREQ 800e6
#define LO_FREQ 90e6
#define THREADS_PER_BLOCK 512
#define BLOCKS_PER_GRID 320 * DOWNSAMPLE
#define BLOCK_SIZE THREADS_PER_BLOCK * BLOCKS_PER_GRID
#define LOOPS 1
#define SIN_TABLE_LENGTH 4096
#define PACKETS_TO_BUFFER BLOCK_SIZE / PAYLOAD_SIZE // number of packets to accumulate before GPU decode
 // how many loops of block size to do

uint64_t num_pkts = 0;
uint64_t missed_heaps  = 0;
uint64_t spead_id_zero = 0;
unsigned int max_received_tv_delta = 0;
uint64_t num_bytes = 0;
uint64_t memory_copies = 0;
snf_ring_t hring;

#define TOGGLE(i) ((i+ 1) & 1)
#define TV_TO_US(tv) ((tv)->tv_sec * 1000000 + (tv)->tv_usec)
unsigned int itvl_idx = 0;
struct itvl_stats {
  struct timeval tv;
  uint64_t usecs;
  uint64_t num_pkts;
  uint64_t num_bytes;
} itvl[2];

float secs_delta;

void
stats()
{
  struct snf_ring_stats stats;
  uint64_t nbytes;
  int rc;
  if ((rc = snf_ring_getstats(hring, &stats))) {
    perror("nic stats failed");
  }

  fprintf(stderr,"\n");
  if (num_pkts == stats.ring_pkt_recv) {
    fprintf(stderr,"Packets received   in HW:        %" PRIu64 "\n", num_pkts);
  } else {
    fprintf(stderr,"Packets received,    app:        %" PRIu64 ", ring: %" PRIu64 "\n",
           num_pkts, stats.ring_pkt_recv);
  }

  fprintf(stderr,"Total bytes received,  app:        %" PRIu64 " (%" PRIu64 " MB)\n",
          num_bytes, num_bytes / 1024 / 1024);
  nbytes = stats.nic_bytes_recv;
  nbytes -= (8 /* HW header */ + 4 /* CRC */) * stats.nic_pkt_recv;
  fprintf(stderr,"Total bytes received + HW aligned: %" PRIu64 " (%" PRIu64 " MB)\n",
          nbytes, nbytes / 1024 / 1024);
  if (num_pkts > 0) {
    fprintf(stderr,"Average Packet Length:    %" PRIu64 " bytes\n",
          num_bytes / num_pkts);
  }

  fprintf(stderr,"Dropped, NIC overflow:    %" PRIu64 "\n", stats.nic_pkt_overflow);
  fprintf(stderr,"Dropped, ring overflow:   %" PRIu64 "\n", stats.ring_pkt_overflow);
  fprintf(stderr,"Dropped, bad:             %" PRIu64 "\n\n", stats.nic_pkt_bad);
  fprintf(stderr,"SPEAD ID of packet 0:     %" PRIu64 "\n", spead_id_zero);
  fprintf(stderr,"Missed SPEAD Packets:     %" PRIu64 "\n\n", missed_heaps);
}

void
print_periodic_stats(void)
{
  struct itvl_stats *this_itvl = &itvl[itvl_idx];
  struct itvl_stats *last_itvl = &itvl[TOGGLE(itvl_idx)];
  float delta_secs;
  uint64_t delta_pkts;
  uint64_t delta_bytes;
  uint32_t pps;
  float gbps;
  float bps;

  gettimeofday(&this_itvl->tv, NULL);
  this_itvl->usecs = TV_TO_US(&this_itvl->tv);
  this_itvl->num_pkts = num_pkts;
  this_itvl->num_bytes = num_bytes;
  delta_secs = (this_itvl->usecs - last_itvl->usecs) / 1000000.0;
  delta_pkts = this_itvl->num_pkts - last_itvl->num_pkts;
  delta_bytes = this_itvl->num_bytes - last_itvl->num_bytes;

  if (delta_pkts != 0) {
    pps = delta_pkts / delta_secs;
    bps = ((float) delta_bytes * 8) / delta_secs;
    gbps = bps / 1000000000.0;

    fprintf
      (stderr,"%" PRIu64 " pkts (%" PRIu64 "B) in %.3f secs (%u pps), Avg Pkt: %"
       PRIu64 ", BW (Gbps): %6.3f\n", delta_pkts, delta_bytes, delta_secs,
       pps, delta_bytes / delta_pkts, gbps);
    fflush(stderr);
  }

  itvl_idx = TOGGLE(itvl_idx);
}

void
sigexit(int sig)
{
  stats();
  exit(0);
}

void
sigalrm(int sig)
{
  print_periodic_stats();
  alarm(5);
}

void
usage(void)
{
 fprintf(stderr,"Usage: kat_simple-recv [-d <ring sz>] [-t] [-b <board number>] [-f <output_filename>] [-p] [-n <num pkts>]\n");
 fprintf(stderr,"                  -t: print periodic statistics\n");
 fprintf(stderr,"                  -b <board number>: Myri10G board number to use.\n");
 fprintf(stderr,"                  -p: poll for packets instead of blocking\n");
 fprintf(stderr,"                  -n <num pkts>: number of packet to receive (default: 0 - infinite)\n");
 fprintf(stderr,"                  -d <ring sz>: size if the recieve ring in bytes\n");
 fprintf(stderr,"                  -f: redirect ot file instead of standard out\n");
 exit(1);
}

uint64_t
host_nsecs(void)
{
  uint64_t nsecs;
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  nsecs = (uint64_t) ts.tv_sec * 1000000000 + ts.tv_nsec;
  return nsecs;
}

// GPU Kernels
__constant__ float cTaps[512];
__constant__ float cSin[SIN_TABLE_LENGTH];
__global__ void mix(char *input_buffer, char *output_buffer, int lo_interval, int sin_table_length)
{
 int idx = blockIdx.x*blockDim.x + threadIdx.x;
 output_buffer[idx] = int(((float)cSin[((sin_table_length/4) + lo_interval*idx) % sin_table_length] * (float)input_buffer[idx]));
}

__global__ void poly_fir(char *input_buffer, float *output_buffer, int no_taps, int upsample, int downsample)
{
 __shared__ char shared_input[13312];
  // this is sized so that (no_taps / upsample) + (poly_interval * (upsample-1)) + (threadIdx.x * downsample) < 16384 - 10
  // for current example this gives 32 + 18 + (256 * 25) = 6450 < 16374
  // The zero position is memory size from the start (e.g. 50 bytes). This represents the prior input for this filter run
  // and should fix boundary issues.
  // No of output samples from this is X * 4. e.g. 256 * 4 = 1024. input_samples / output_samples = 6.25 = 25 / 4 = downsample / upsample
  // For expedience we get each thread to contribute a single float to the front of the buffer to make the memory_size segment.
  // This is redundant by blockDim.x - memory_size. So our size becomes 256 * 26 = 665
 int memory_size = blockDim.x; //(no_taps / upsample) + (poly_interval * (upsample-1));

 if (blockIdx.x > 0) shared_input[threadIdx.x] = input_buffer[blockIdx.x*blockDim.x*downsample + threadIdx.x - blockDim.x];
 else shared_input[threadIdx.x] = 0;
 int idx = blockIdx.x*blockDim.x + threadIdx.x;

 for (int k=0; k < downsample; k++) {
  shared_input[memory_size+threadIdx.x*downsample+k] = input_buffer[blockIdx.x*blockDim.x*downsample + threadIdx.x*downsample + k];
 }

 __syncthreads();
 float temp_output = 0;
 int fir_idx = (threadIdx.x*downsample + memory_size);
#pragma unroll 128
 for (int i=0; i < no_taps; i++) {
  temp_output += cTaps[i] * (float)shared_input[fir_idx - i];
 }
 output_buffer[blockIdx.x*blockDim.x + threadIdx.x] = temp_output;
}

__global__ void float_cast(float *in, char *out)
{
 int idx = blockDim.x*blockIdx.x + threadIdx.x;
 out[idx] = (char) in[idx];
}

// end of GPU Kernels

int
main(int argc, char **argv)
{
  int rc;
  snf_handle_t hsnf;
  struct snf_recv_req recv_req;
  char c;
  int periodic_stats = 0;
  int boardnum = 0;
  uint64_t pkts_expected = 0xffffffffffffffffULL;
  int open_flags = 0;
  uint64_t dataring_sz = 0;
  int timeout_ms = -1;
// SPEAD counters
  unsigned short spead_header_size = 0;
  uint64_t heap_cnt = 0;
  uint64_t old_heap_cnt = 0;

// GPU related
  float et=0;
  cudaEvent_t t_start, t_stop;
  float elapsedTimeInMs = 0.0f;
  int buffer_size = BLOCK_SIZE;
  int packets_to_buffer = PACKETS_TO_BUFFER;
  int sample_rate = SAMPLE_FREQ;
   // our adc sampling frequency
  int upsample = 1;
  int downsample = DOWNSAMPLE;
   // coefficients to sort out the output sample rate
   // in this case giving us 128 MHz sampling
  int sin_table_length = SIN_TABLE_LENGTH;
  int lo_freq = LO_FREQ;
   // the number of samples in the sin lookup table
  int sin_table_size = sizeof(float) * sin_table_length;
  int lo_interval = int(((float)sin_table_length / sample_rate) * lo_freq);
   // the stepping interval through the lo sin table. May result in slightly different lo_freq from that
  int no_taps = 0;
   // number of filter taps. Calculated once filter data is loaded.
  int no_output_samples = int(((float)buffer_size / downsample));
   // overall number of output samples to produce for this block

  int fh,outfh;
  char *fir_taps_file = "./fir_16m_128tap.dat";
  char *output_filename = NULL;
  struct stat stat_buf;
 
  float *fir_taps;
  char *host_char_buffer;
  float *upsample_buffer;
  float *sin_table;
   // host buffers

  char *device_char_buffer;
  float *device_fir_taps;
  float *device_output_buffer;
  float *device_upsample_buffer;
  float *device_float_buffer;
  float *device_fir_buffer;
   // device buffers
  char *dst;

  /* get args */
  while ((c = getopt(argc, argv, "vtb:f:pn:d:S:")) != -1) {
    if (c == 't') {
      periodic_stats = 1; 
    } else if (c == 'b') {
      boardnum = strtoul(optarg, NULL, 0);
    } else if (c == 'f') {
      output_filename = optarg; 
    } else if (c == 'p') {
      timeout_ms = 0;
    } else if (c == 'n') {
      pkts_expected = strtoul(optarg, NULL, 0);
    } else if (c == 'd') {
      dataring_sz = strtoull(optarg, NULL, 0);
    } else {
      fprintf(stderr,"Unknown option: %c\n", c);
      usage();
    }
  }

  snf_init(SNF_VERSION_API);
  rc = snf_open(boardnum, 1, NULL, dataring_sz, open_flags, &hsnf);
  if (rc) {
    errno = rc;
    perror("Can't open snf for sniffing");
    return -1;
  }
  rc = snf_ring_open(hsnf, &hring);
  if (rc) {
    errno = rc;
    perror("Can't open a receive ring for sniffing");
    return -1;
  }
  rc = snf_start(hsnf);
  if (rc) {
    errno = rc;
    perror("Can't start packet capture for sniffing");
    return -1;
  }

  fprintf(stderr,"Packet buffer holds %i packets. (%i bytes)\n", packets_to_buffer, packets_to_buffer * PAYLOAD_SIZE);
  fprintf(stderr,"snf_recv ready to receive\n");

  if (SIG_ERR == signal(SIGINT, sigexit))
    exit(1);
  if (SIG_ERR == signal(SIGTERM, sigexit))
    exit(1);

  if (periodic_stats) {
    if (SIG_ERR == signal(SIGALRM, sigalrm))
      exit(1);
    itvl[itvl_idx].num_pkts = 0;
    itvl[itvl_idx].num_bytes = 0;
    gettimeofday(&itvl[itvl_idx].tv, NULL);
    itvl[itvl_idx].usecs = 0;
    itvl_idx = TOGGLE(itvl_idx);
    alarm(5);
  }

// Start of GPU config
 fprintf(stderr,"Actual lo freq is: %f MHz (%i, %i, %i, %i)\n", lo_interval / ((float)sin_table_length / sample_rate), lo_interval, sin_table_length, sample_rate, lo_freq);
 fprintf(stderr,"Producing %i output samples per block (%i samples).\n",no_output_samples,buffer_size);

  // gpu related malloc...
 host_char_buffer = (char*)malloc(buffer_size);
 upsample_buffer = (float*)malloc(sizeof(float) * upsample * (buffer_size + THREADS_PER_BLOCK));
 sin_table = (float*)malloc(sin_table_size);
 memset(host_char_buffer, (char) 0, no_output_samples);
 memset(upsample_buffer, (char) 0, sizeof(float) * upsample * (buffer_size + THREADS_PER_BLOCK));
  // zero as we use part of this for our initial zero padding block

 if (output_filename != NULL) { 
  fprintf(stderr,"Writing output data to :%s:\n", output_filename);
  outfh = open(output_filename, O_RDWR | O_CREAT, S_IRWXU);
  if (outfh < 0) {
   perror("Failed to open data file");
   exit(1);
  }
 }
 else {
  if (isatty(1)) { 
   fprintf(stderr,"You have requested data be sent to stdout which has not been redirected. This is a bad idea :). Quitting...\n");
   exit(1);
  }
  outfh = 1;
 }

 fprintf(stderr,"\nOutput filehandle is %i\n", outfh);

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

 cudaEventCreate(&t_start);
 cudaEventCreate(&t_stop);

 cudaMalloc((void**)&device_char_buffer, buffer_size);
  // device buffer with space for initial zero padding
 cudaMalloc((void**)&device_output_buffer, sizeof(float) * no_output_samples);
 cudaMalloc((void**)&device_float_buffer, sizeof(float) * buffer_size);
 cudaMalloc((void**)&device_upsample_buffer, sizeof(float) * upsample * (buffer_size + THREADS_PER_BLOCK));
 cudaMalloc((void**)&device_fir_buffer, sizeof(float) * upsample * buffer_size);
 cudaMalloc((void**)&device_fir_taps, sizeof(float) * no_taps);
  // allocate the device storage

 fprintf(stderr,"Copying FIR taps and SIN lookup table to GPU...\n");
 cudaMemcpy(device_fir_taps, fir_taps, sizeof(float) * no_taps, cudaMemcpyHostToDevice);
  // copy the filter taps to the device
 //cudaMemcpyToSymbol(cTaps, fir_taps, sizeof(float) * no_taps);
 //cudaMemcpyToSymbol(cSin, sin_table, sin_table_size);

 cudaMemcpy(device_upsample_buffer, upsample_buffer, sizeof(float) * upsample * buffer_size, cudaMemcpyHostToDevice);
 cudaMemcpy(device_output_buffer, upsample_buffer, sizeof(float) * no_output_samples, cudaMemcpyHostToDevice);
 cudaMemcpy(device_fir_buffer, upsample_buffer, sizeof(float) * upsample * buffer_size, cudaMemcpyHostToDevice);
 cudaMemcpy(device_float_buffer, upsample_buffer, sizeof(float) * buffer_size, cudaMemcpyHostToDevice);

 fprintf(stderr,"GPU Configuration: blocks per grid: %i, threads per block: %i\n",BLOCKS_PER_GRID, THREADS_PER_BLOCK);
    // polyphase method
 dim3 threads(THREADS_PER_BLOCK / upsample, upsample);
   // the downsample spaced blocks are indexed by thread.x and the upsample number of fir filters are indexed by y
 dim3 blocks(BLOCKS_PER_GRID / downsample,1);
// end of GPU config

  while (num_pkts < pkts_expected) {
    rc = snf_ring_recv(hring, timeout_ms, &recv_req);
    if (rc == EAGAIN || rc == EINTR)
      continue;
    else if (rc == 0) {
      if ((unsigned short)((unsigned char *)recv_req.pkt_addr)[UDP_HDRLEN] == 83) {
       spead_header_size = ((unsigned short)((unsigned char *)recv_req.pkt_addr)[UDP_HDRLEN + 7] + 1) * 8; 
       heap_cnt = (uint64_t)((unsigned char *)recv_req.pkt_addr)[UDP_HDRLEN+8+3] * 256 * 256 * 256 * 256 + (uint64_t)((unsigned char *)recv_req.pkt_addr)[UDP_HDRLEN+8+4] * 256 * 256 * 256 + ((unsigned char *)recv_req.pkt_addr)[UDP_HDRLEN+8+5] * 256 * 256  + ((unsigned char *)recv_req.pkt_addr)[UDP_HDRLEN+8+6] * 256 + ((unsigned char *)recv_req.pkt_addr)[UDP_HDRLEN+8+7];
       if (num_pkts > 0 && heap_cnt > old_heap_cnt + 1 ) {
        missed_heaps += (heap_cnt - old_heap_cnt) - 1;
        fprintf(stderr,"Heap cnt diff:%lu for heap:%lu in packet: %lu (header: %i)\n",heap_cnt - old_heap_cnt, heap_cnt, num_pkts, ((unsigned char *)recv_req.pkt_addr)[UDP_HDRLEN]);
	//sigexit(0);
       }
       old_heap_cnt = heap_cnt;
       memcpy(host_char_buffer + ((num_pkts % packets_to_buffer) * PAYLOAD_SIZE), (unsigned char *)recv_req.pkt_addr+UDP_HDRLEN+spead_header_size, recv_req.length - UDP_HDRLEN - spead_header_size);

       if (num_pkts % 100000 == 0) fdatasync(outfh);
       if (num_pkts % 1000000 == 0) { 
	if (num_pkts == 0) spead_id_zero = heap_cnt;
        if (periodic_stats) {
         fprintf(stderr,"Packet: %" PRIu64 ", Heap count: %" PRIu64 ", Differential (heap_cnt - first_heap_cnt - num_pkts): %li .\n",num_pkts, heap_cnt, (long)(heap_cnt - spead_id_zero) - num_pkts);
         fprintf(stderr,"Accumulated GPU copy time for size %i is %f in %lu writes (avg: %f, rate: %f MBps, prb: %i)\n", buffer_size, elapsedTimeInMs, memory_copies, elapsedTimeInMs / (float)memory_copies, ((long int)memory_copies * buffer_size) / (float)(elapsedTimeInMs*1000), packets_to_buffer);
        }
        //fdatasync(outfh);
       }
       if (num_pkts % packets_to_buffer == 0 && num_pkts > 0) { 
	// ***** Processing *********
        cudaEventRecord(t_start, 0);

        cudaMemcpy(device_char_buffer, host_char_buffer, buffer_size, cudaMemcpyHostToDevice);
         // need to recal lo_offset each loop

        mix<<<BLOCKS_PER_GRID / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(device_char_buffer, device_char_buffer, lo_interval, sin_table_length);

        poly_fir<<<blocks, threads>>>(device_char_buffer, device_output_buffer, no_taps, upsample, downsample);

        float_cast<<<no_output_samples/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(device_output_buffer, device_char_buffer);
        cudaMemcpy(host_char_buffer, device_char_buffer, no_output_samples, cudaMemcpyDeviceToHost);

        cudaEventRecord(t_stop, 0);
        cudaEventSynchronize(t_stop);
        cudaEventElapsedTime(&et, t_start, t_stop);
        memory_copies++;
        elapsedTimeInMs += et;
        // ***** End of Processing *******

	// write results to disk

	lseek(outfh, no_output_samples * memory_copies - 1, SEEK_SET);
        write(outfh,"",1);
        dst = (char *)mmap(NULL, no_output_samples, PROT_READ | PROT_WRITE, MAP_SHARED, outfh, no_output_samples * (memory_copies-1));        
        if (dst == MAP_FAILED) { perror("Memory map failed"); exit(1); }
        memcpy(dst, host_char_buffer, no_output_samples);
        munmap(dst, no_output_samples);
       }
      num_pkts++;
      num_bytes += recv_req.length;
      } // valid SPEAD packet
    }
    else {
      fprintf(stderr, "error: snf_recv = %d (%s)\n",
                 rc, strerror(rc));
      break;
    }
  }

  rc = snf_ring_close(hring);
  if (rc) {
    errno = rc;
    perror("Can't close receive ring");
    return -1;
  }
  rc = snf_close(hsnf);
  if (rc) {
    errno = rc;
    perror("Can't close sniffer device");
    return -1;
  }
  stats();
  return 0;
}

struct pkt_hdr {
  uint32_t length;
  uint32_t ofst;
};
typedef struct pkt_hdr pkt_hdr_t;
