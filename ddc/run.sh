#!/bin/bash
#
# This is an example script
#
#These commands set up the Grid Environment for your job:
#PBS -N SKA__DDC
#PBS -l nodes=srvslsgpu001:ppn=1:seriesGPU,walltime=00:59:00
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
exeFile=/home/bhugo/ska-res/ska-ddc/ddc/build/gpu_ddc
inputSignal=/home/bhugo/ska-res/ska-ddc/ddc/pytone.dat
filterFile=/home/bhugo/ska-res/ska-ddc/ddc/fir_16m_128tap.dat
outputFile=/home/bhugo/ska-res/ska-ddc/ddc/gpu_ddc_out.dump
$exeFile $inputSignal $filterFile $outputFile
