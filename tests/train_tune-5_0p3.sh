#!/bin/sh
#
# Model training with a mean <pile-up> = 60
#
# m.mieskolainen@imperial.ac.uk, 2023

source setenv.sh

# Set system memory limits
ulimit -s unlimited # stack
ulimit -v unlimited # virtual memory

python src/trackml_train.py \
	--param tune-5 \
	--cluster "transformer" \
	--soft_reset 0 \
	--learning_rate 5e-4 \
	--learning_rate_pdf 0 \
	--scheduler_type "warm-cos" \
	--optimizer "AdamW" \
	--epoch -1 \
	--save_tag f-0p3-hyper-5 \
	--rfactor_start 0.3 --rfactor_end 0.3 --noise_ratio 0.05 \
	--node2node hyper --ncell 524288 \
	--validate 0 \
	--batch_size 1 \
	--fp_dtype "float32"
