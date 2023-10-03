#!/bin/sh
#
# Technical tests of the Voxel-Dynamics graph constructor
#
# m.mieskolainen@imperial.ac.uk, 2023

source setenv.sh

# Set system memory limits
ulimit -s unlimited # stack
ulimit -v unlimited # virtual memory

# A fixed pile-up mean and changing the number of V cells
for NCELL in 65536 131072 262144
do
	python src/trackml_inference.py \
		--param tune-5 \
		--pre_only 1 \
		--epoch -1 \
		--read_tag f-0p1-hyper-5 \
		--event_start 8750 --event_end 8750 \
		--rfactor 0.1 --noise_ratio 0.05 \
		--node2node hyper \
		--ncell $NCELL
		#--device cpu
done

# A fixed number of V cells and changing the pile-up mean
for RFACTOR in 0.1 0.3 0.5 1.0
do
	python src/trackml_inference.py \
		--param tune-5 \
		--pre_only 1 \
		--epoch -1 \
		--read_tag f-0p1-hyper-5 \
		--event_start 8750 --event_end 8750 \
		--rfactor $RFACTOR --noise_ratio 0.05 \
		--node2node hyper \
		--ncell 524288
		#--device cpu
done
