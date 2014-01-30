#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "inv_pfb.h"

uint32_t readDataFromDisk(const char * filename, uint32_t element_size, uint32_t length, void * buffer, uint32_t seek = 0);
uint32_t writeDataToDisk(const char * filename, uint32_t element_size, uint32_t length, const void * buffer, bool blankfile = false);

/**Reads data from disk
 @args filename
 @args element_size size of the individual elements in bytes
 @args length number of elents to read
 @args buffer pre-allocated buffer of length 'element_size*length' in bytes
 @returns 0 if file could not be opened or seek failed, else number of elements of size "element_size" read
*/
uint32_t readDataFromDisk(const char * filename, uint32_t element_size, uint32_t length, void * buffer, uint32_t seek){
        FILE * hnd = fopen(filename,"r");
        uint32_t elemsRead = 0;
        if (hnd != NULL){
		if (seek != 0){
                        if (fseek(hnd,seek,SEEK_SET) != 0)
                                return 0;
                }
                elemsRead = fread(buffer,element_size,length,hnd);
                fclose(hnd);
        }
        return elemsRead;
}

/**Writes data to disk
 @args filename
 @args element_size size of the individual elements in bytes
 @args length number of elents to write
 @args buffer pre-allocated buffer of length 'element_size*length' in bytes
 @returns 0 if file could not be opened, else number of elements of size "element_size" successfully written to disk
*/
uint32_t writeDataToDisk(const char * filename, uint32_t element_size, uint32_t length, const void * buffer, bool blankfile){
	FILE * hnd = fopen(filename, blankfile ? "w" : "a");
	uint32_t elemsWrote = 0;
        if (hnd != NULL){
                elemsWrote = fwrite(buffer,element_size,length,hnd);
                fclose(hnd);
        }
	return elemsWrote;
}

/**
TODO: THIS IS FOR DEBUGGING PURPOSES ONLY... the real input data should not have redundant samples!
*/

void cutDuplicatesFromData(const complex_float * in, complex_float * out, uint32_t orig_count){
        assert(orig_count % N == 0); //only whole blocks should be processed
        for (uint32_t i = 0; i < orig_count; i += N){
                memcpy(out + (i/N)*(N/2 + 1), in + i, (N/2 + 1) * sizeof(complex_float));
        }
}

int main ( int argc, char ** argv ){
        char * taps_filename;
        char * pfb_output_filename;
        char * output_filename;
        uint32_t num_samples;
        if (argc != 5){
                fprintf(stderr, "expected arguements: 'prototype_filter_file' 'no_complex_to_read' 'pfb_output_file' 'output_file'\n");
                return 1;
        }
        taps_filename = argv[1];
        num_samples = atoi(argv[2]);
	assert(num_samples % N == 0); //TODO: this is for debugging only, it should become N/2 + 1 when dealing with actual pfb data
        pfb_output_filename = argv[3];
        output_filename = argv[4];
        printf("Performing operation on %d samples from '%s'\n",num_samples,pfb_output_filename);

        //Read in taps file (prototype filter)  
        float * taps = (float*) malloc(sizeof(float)*WINDOW_LENGTH);
        if (readDataFromDisk(taps_filename,sizeof(float),WINDOW_LENGTH,taps) != WINDOW_LENGTH){
                fprintf(stderr, "Prototype filter has to be %d long\n",WINDOW_LENGTH);
                return 1;
        }

        //Read in input file (output of earlier pfb process, so these will be complex numbers
        complex_float * pfb_data = (complex_float*) malloc(sizeof(complex_float)*(num_samples)); //pad the array with N*P positions
        if (readDataFromDisk(pfb_output_filename,sizeof(complex_float),num_samples,pfb_data) != num_samples){
                fprintf(stderr, "input pfb data does not contain %d samples\n", num_samples);
                return 1;
	}
        uint32_t trimmed_input_length = (N/2+1)*(num_samples / N);
        complex_float * trimmed_pfb_data = (complex_float*) malloc(sizeof(complex_float)*trimmed_input_length);
        cutDuplicatesFromData(pfb_data,trimmed_pfb_data,num_samples);

        //Setup the device and copy the taps
        initDevice(taps);
        float * output = (float*) malloc(sizeof(float)*num_samples);

        //do some processing
	printf("\033[0;31mProcessing starting\033[0m\n---------------------------------------------------------\n%d non-redundant samples in the input data\n",trimmed_input_length);
	uint32_t num_loops = (uint32_t)ceil(trimmed_input_length / float(LOOP_LENGTH));
	printf("Only %d samples can be processed by the GPU at a time. %d iterations needed to complete the inverse pfb\n",LOOP_LENGTH,num_loops);
	printf("---------------------------------------------------------\n");
	for (uint32_t i = 0; i < num_loops; ++i){
		uint32_t num_blocks_to_process = ((uint32_t)fmin(trimmed_input_length - i * LOOP_LENGTH,LOOP_LENGTH)) / FFT_SIZE; 
		printf("\033[0;32mExecuting loop %d/%d. Processing %d blocks of size %d complex numbers\033[0m\n\n",i+1,num_loops,num_blocks_to_process,FFT_SIZE);
	        processNextStride(trimmed_pfb_data + (i * LOOP_LENGTH), output, num_blocks_to_process);
        	writeDataToDisk(output_filename,sizeof(float),num_blocks_to_process*N,output,i == 0);
	}
	printf("---------------------------------------------------------\n");
        //release the device
        releaseDevice();

        //free any hostside memory:
        free(output);
        free(pfb_data);
        free(taps);
        free(trimmed_pfb_data);
        printf("\033[1;33mApplication Terminated Normally\033[0m\n---------------------------------------------------------\n");
}
