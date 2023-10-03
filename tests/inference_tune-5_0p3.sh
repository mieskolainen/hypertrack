#!/bin/sh
#
# Inference test for <pile-up> = 60 (rfactor = 0.3)
# using pile-up matched training
#
# m.mieskolainen@imperial.ac.uk, 2023

source setenv.sh

# Set system memory limits
ulimit -s unlimited # stack
ulimit -v unlimited # virtual memory

python src/trackml_inference.py \
	--param tune-5 \
	--cluster "transformer" \
	--epoch -1 \
	--event_start 8750 --event_end 9999 \
	--read_tag f-0p3-hyper-5 \
	--out_tag  f-0p3-hyper-5 \
	--rfactor 0.3 --noise_ratio 0.05 \
	--node2node hyper --ncell 524288 \
	--fp_dtype "float32"
