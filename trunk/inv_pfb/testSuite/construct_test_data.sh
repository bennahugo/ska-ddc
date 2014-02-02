#!/bin/bash
#tweak these variables if you must:
N=512
P=8
num_samples_to_use=$((N * P * 105))
output_directory=../data_out
tone_type=noise

#don't tweak these formulas unless essential:
#N/2 + 1 non-redundant samples in the fft output of the pfb, by the Hermite-symmetric property of real FFTs, BUT: discard last one due to the SKA infrastructure:
non_redundant_samples=$((N / 2))
non_redundant_pfb_output=$(( (num_samples_to_use / N) * non_redundant_samples ))

if [ -d "$output_directory" ]; then
		rm -r $output_directory		
fi

mkdir $output_directory
stats="Test data characteristics:\n--------------------------------------\nTone type: $tone_type\nN: $N\nP: $P\nNumber of samples in tone: $num_samples_to_use\nNumber of non-redundant samples in pfb output: $non_redundant_pfb_output"
echo -e "--------------------------------------\nDumping test data into: $output_directory\n$stats"

#invoke scripts
python tone_generator.py $output_directory/noise.dat $num_samples_to_use $tone_type $N $P
python filter_generator.py $output_directory/prototype_FIR.dat $N $P
python pfb_generator.py $output_directory/noise.dat $output_directory/prototype_FIR.dat $output_directory/pfb.dat $output_directory/unfiltered_ffts.dat $num_samples_to_use $N $P
python ipfb_generator.py $output_directory/pfb.dat $output_directory/prototype_FIR.dat $output_directory/py_ifftedPFB.dat $output_directory/py_ipfb.dat $non_redundant_pfb_output $N $P
echo -e "$stats" > $output_directory/test_data_characteristics.txt

