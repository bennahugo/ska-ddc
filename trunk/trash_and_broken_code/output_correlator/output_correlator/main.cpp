#include <iostream>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <omp.h>
#include "timer.h"

/*
 * Cross correlation is embarisingly parallel therefore we can simply parallelize the outer loop
 * 
 * This program assumes two signals are of equal length
 * 
 * !!!
 * Note: that 8-bit samples can have a value range of 256 unique values. We need to make sure that
 * floor(log_2(256 * (number of samples)))+1 bits are available to store the correlated value for each
 * output. The user of this program should ensure this bit count is no bigger than 64 bits
 * !!!
 */



int32_t loadInput(int8_t * buffer, char * filename, uint32_t numBytesToRead){
  using namespace std;
  FILE * hnd = fopen(filename,"r");
  if (hnd != NULL){
    int32_t bytesRead = std::fread(buffer,sizeof(int8_t),numBytesToRead,hnd);
    fclose(hnd);
    return bytesRead;
  } else return 0;
}
__attribute__((optimize("unroll-loops")))
void cross_correlate(int8_t * a, int8_t * b, int64_t * out, uint32_t size_a, uint32_t size_b){
  assert(size_a != 0 && size_b != 0);
  for (uint32_t i = 0; i < size_a; ++i){  
    int64_t outval = 0;
    for (uint32_t j = 0; j < size_b; ++j){
      int32_t idx = i + j;
      if (idx < size_a)
	outval += b[j] * (a[idx]);
    }
    out[i] = outval;
    
    if (i % uint32_t(size_a*0.05) == i / size_a) 
      printf("CROSS CORRELATION AT %f \%\n",i / float(size_a) * 100.0f);
  } 
}

__attribute__((optimize("unroll-loops")))
void parallel_cross_correlate(int8_t * a, int8_t * b, int64_t * out, uint32_t size_a, uint32_t size_b){
  assert(size_a != 0 && size_b != 0);
#pragma omp parallel for
  for (uint32_t i = 0; i < size_a; ++i){  
    int64_t outval = 0;
    for (uint32_t j = 0; j < size_b; ++j){
      int32_t idx = i + j;
      if (idx < size_a)
	outval += b[j] * (a[idx]);
    }
    out[i] = outval;
  } 
}


int main(int argc, char **argv) {
    using namespace std;
    if (argc != 4){
	printf("Please supply 'numSamples' 'input stream 1 file name' 'input stream 2 file name'");
	return 1;
    }
    uint32_t numSamples = atoi(argv[1]);
    char * stream1File = argv[2];
    char * stream2File = argv[3];
    printf("Allocing memory for streams\n");
    int8_t * stream1 = (int8_t *)malloc(numSamples * sizeof(int8_t));
    int8_t * stream2 = (int8_t *)malloc(numSamples * sizeof(int8_t));
    printf("Reading stream '%s'\n",stream1File); 
    uint32_t bytesRead = loadInput(stream1,stream1File,numSamples);
    if (bytesRead != numSamples){
      fprintf(stderr,"Stream '%s' does not contain the required %d samples, but only contains %d. Giving up.\n",stream1File,numSamples,bytesRead);
      return 1;
    }
    printf("Reading stream '%s'\n",stream2File);
    bytesRead = loadInput(stream2,stream2File,numSamples);
    if (bytesRead != numSamples){
      fprintf(stderr,"Stream '%s' does not contain the required %d samples, but only contains %d. Giving up.\n",stream2File,numSamples,bytesRead);
      return 1;
    }
    printf("Streams '%s' and '%s' of size %d read successfully\n",stream1File,stream2File,numSamples);
    
    /*
    printf("Starting cross correlation\n");
    int64_t * correlation = (int64_t*)malloc(sizeof(int64_t)*numSamples);
    timer::tic();
    cross_correlate(stream1,stream2,correlation,numSamples,numSamples);
    printf("Completed cross correlation in %f seconds\n",timer::toc());
    */
    
    printf("Starting parallel cross correlation with %d threads\n",omp_get_max_threads());
    int64_t * par_correlation = (int64_t*)malloc(sizeof(int64_t)*numSamples);
    timer::tic();
    parallel_cross_correlate(stream1,stream2,par_correlation,numSamples,numSamples);
    printf("Completed parallel cross correlation in %f seconds\n",timer::toc());
    
  /*  
    //DEBUG::
    printf("SANITY CHECK ");
    for (uint32_t i = 0; i < numSamples; ++i)
      if (correlation[i] != par_correlation[i]){
	printf("<FAILED @ ELEMENT %d>\n",i);
	return 1;
      }
    printf("<PASS>\n");
    */
    //find the stream shift
    int64_t maxV = par_correlation[0];
    int64_t maxA = 0;
    for (uint32_t i = 1; i < numSamples; ++i)
      if (par_correlation[i] > maxV){
      maxV = par_correlation[i];
      maxA = i;
    }
    printf("Stream '%s' offset from stream '%s' by %d samples\n",stream2File,stream1File,maxA);
    printf("Program terminated normally\n");
    //finally free allocated memory:
    free(stream1);
    free(stream2);
    free(par_correlation);
    //free(correlation);
    return 0;
}
