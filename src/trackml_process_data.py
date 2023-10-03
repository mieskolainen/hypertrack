# Process Kaggle ML tracking challenge dataset into pickle files
#
# m.mieskolainen@imperial.ac.uk, 2023

import sys
sys.path.append("../hypertrack")

import os
import multiprocessing
import pickle
from argparse import ArgumentParser

from hypertrack import trackml


def task(event, args):
    CWD = os.getcwd()
    
    idstr = f'event{event:09d}'
    PATH = args.path + '/' + idstr

    print(f'Processing file: {PATH}')
    obj = trackml.process_data(PATH=PATH, verbose=args.verbose)
    
    filename = f'{CWD}/data/trackml_{idstr}.pkl'
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved to: {filename}')

def main():
    CWD = os.getcwd()
    
    if not os.path.exists(f'{CWD}/data'):
        os.makedirs(f'{CWD}/data')

    parser = ArgumentParser()
    parser.add_argument("-p",  "--path",        action="store", type=str,      default="../trackml/train_1", help="TrackML dataset path")
    parser.add_argument("-e0", "--event_start", action="store", default=1000,  type=int, help="Event file start")
    parser.add_argument("-e1", "--event_end",   action="store", default=2399,  type=int, help="Event file end")
    parser.add_argument("-v",  "--verbose",     action="store", default=0,     type=int, help="Verbose mode")
    
    args = parser.parse_args()
    print(args)

    # Use all available cores, otherwise specify the number you want as an argument
    pool = multiprocessing.Pool()
    for i in range(args.event_start, args.event_end+1):
        pool.apply_async(task, args=(i, args))
    
    pool.close()
    pool.join()
    
    print('Done', flush=True)

if __name__ == '__main__':
    main()
