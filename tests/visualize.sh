#!/bin/sh
#
# Run with: source tests/test_visualize.sh
#
# Process one event and do visualization
#
# m.mieskolainen@imperial.ac.uk, 2023

python src/trackml_process_data.py --path '../trackml/train_1' -e0 1000 -e1 1000
python src/trackml_visualize.py
