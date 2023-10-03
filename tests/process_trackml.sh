#!/bin/sh
#
# Process TrackML data into pickle files
#
# m.mieskolainen@imperial.ac.uk, 2023

python src/trackml_process_data.py --path '../trackml/train_1' -e0 1000 -e1 2399

python src/trackml_process_data.py --path '../trackml/train_1' -e0 2450 -e1 2819
python src/trackml_process_data.py --path '../trackml/train_2' -e0 2820 -e1 4589
python src/trackml_process_data.py --path '../trackml/train_3' -e0 4590 -e1 5899

python src/trackml_process_data.py --path '../trackml/train_3' -e0 5950 -e1 6409
python src/trackml_process_data.py --path '../trackml/train_4' -e0 6410 -e1 8179
python src/trackml_process_data.py --path '../trackml/train_5' -e0 8180 -e1 8699

python src/trackml_process_data.py --path '../trackml/train_5' -e0 8750 -e1 9999
