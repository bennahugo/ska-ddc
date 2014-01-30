#!/bin/sh
#
# This is an example script
#
#These commands set up the Grid Environment for your job:
#PBS -N SKA__PFB
#PBS -l nodes=srvslsgpu002:ppn=1:seriesGPU,walltime=00:01:00
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
/home/bhugo/ska-res/ska-ddc/inv_pfb/build/inv_pfb /home/bhugo/ska-res/ska-ddc/inv_pfb/data_out/prototype_FIR.dat 12288 /home/bhugo/ska-res/ska-ddc/inv_pfb/data_out/pfb.dat /home/bhugo/ska-res/ska-ddc/inv_pfb/pfb_inv_noise_c_ver.dat 