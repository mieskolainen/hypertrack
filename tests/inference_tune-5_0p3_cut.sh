#!/bin/sh
#
# Inference test for <pile-up> = 60 (rfactor = 0.3)
# using pile-up matched training [CUT ONLY, no Transformer]
#
# m.mieskolainen@imperial.ac.uk, 2023

source setenv.sh

# Set system memory limits
ulimit -s unlimited # stack
ulimit -v unlimited # virtual memory

python src/trackml_inference.py \
	--param tune-5 \
	--cluster "cut" \
	--epoch -1 \
	--event_start 8750 --event_end 9000 \
	--read_tag f-0p3-hyper-5 \
	--out_tag  f-0p3-hyper-5-cut \
	--rfactor 0.3 --noise_ratio 0.05 \
	--node2node hyper --ncell 524288 \
	--fp_dtype "float32"
