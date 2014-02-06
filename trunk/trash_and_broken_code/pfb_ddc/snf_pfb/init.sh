#!/bin/bash
# may need to rmmod and modprobe megaraid_sas

export LD_LIBRARY_PATH=/home/kat/snf/lib/:$LD_LIBRARY_PATH
export SNF_DATARING_SIZE=4294967296
sudo bash -c "echo 100 > /proc/irq/56/smp_affinity"
