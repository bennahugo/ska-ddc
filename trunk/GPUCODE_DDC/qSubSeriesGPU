#!/bin/sh
#
# This is an example script
#
#These commands set up the Grid Environment for your job:
#PBS -N SKA__DDC
#PBS -l nodes=1:ppn=16:seriesGPU,walltime=00:01:00
#PBS -q GPUQ
#

# Change to the directory from which the job was submitted.  
cd $PBS_O_WORKDIR

# Print the date and time
echo "Script started at "$(date)

# Host name of the node we are executing on
echo ""
echo "Running on: $(hostname)"
echo "-----------------------------------------------------------------------"
/home/bhugo/ska-res/ska-ddc/GPUCODE_DDC/build/gpu_ddc /home/bhugo/ska-res/ska-ddc/GPUCODE_DDC/pytone.dat /home/bhugo/ska-res/ska-ddc/GPUCODE_DDC/fir_16m_128tap.dat /home/bhugo/ska-res/ska-ddc/GPUCODE_DDC/output.dump
