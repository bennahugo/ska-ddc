#!/bin/sh
/usr/local/cuda/bin/nvcc -O kat_gpu_ddc.cu -o kat_gpu_ddc -lpthread -L ../lib/ -lpcap -lsnf -I ../include/
