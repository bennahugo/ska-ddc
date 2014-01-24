#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define N 512; //Number of FFT samples
#define P 8; //Number of Filterbanks
#define WINDOW_LENGTH N*P

struct complex_float {
	float r;
	float i;
};

/**
 Reads data from disk
 @args filename
 @args element_size size of the individual elements in bytes
 @args length number of elents to read
 @args buffer pre-allocated buffer of length 'element_size*length' in bytes
 @returns 0 if file could not be opened, else number of elements of size "element_size" read
*/
uint32_t readDataFromDisk(const char * filename, uint32_t element_size, uint32_t length, float * buffer){
	FILE * hnd = fopen(filename,"r");
	uint32_t bytesRead = 0;
	if (hnd != NULL){
		bytesRead = fread(buffer,element_size,length,hnd);	
		fclose(hnd);
	}
	return bytesRead / element_size;
}

int main ( int argc, char ** argv ){
	using namespace std;
	char * taps_filename;
	char * pfb_output_filename;
	char * output_filename;
	uint32_t num_samples;
	if (argc != 5){
		fprintf(stderr, "expected arguements: 'prototype_filter_file' 'no_complex_to_read' 'pfb_output_file' 'output_file'");
		return 1;
	}	
	taps_filename = argv[1];
	num_samples = atoi(argv[2]);
	pfb_output_filename = arg[3];
	output_filename = arg[4];
	
	/****************************************************************************************************************
	 Read in taps file (prototype filter)
	*****************************************************************************************************************/
	float * taps = new float[WINDOW_LENGTH];
	if (readDataFromDisk(taps_filename,sizeof(float),WINDOW_LENGTH,taps) != WINDOW_LENGTH){
		fprintf(stderr, "Prototype filter has to be %d long",WINDOW_LENGTH);
		return 1;
	}
	/****************************************************************************************************************
	 Read in input file (output of earlier pfb process, so these will be complex numbers
        *****************************************************************************************************************/
	complex_float * pfb_data = new complex_float[num_samples];
	if (readDataFromDisk(pfb_output_filename,sizeof(complex_float),num_samples,pfb_data) != num_samples){
		fprintf(stderr, "input pfb data does not contain the %d of samples", num_samples);
		return 1;
	}	

	delete[] pfb_data;
	delete[] taps;	
}
