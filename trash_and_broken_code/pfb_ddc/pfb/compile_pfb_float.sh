#!/bin/sh
/usr/local/cuda/bin/nvcc -g pfb_float.cu -o pfb_float -lcufft
