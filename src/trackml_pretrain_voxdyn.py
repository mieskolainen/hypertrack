# HyperTrack Voxel-Dynamics (VD) graph constructor trainer
#
# m.mieskolainen@imperial.ac.uk, 2023

import sys
sys.path.append("../hypertrack")

import os
import numba
import numpy as np
from argparse import ArgumentParser

from hypertrack import voxdyn, trackml, iotools


def main():
    CWD = os.getcwd()
    numba.config.THREADING_LAYER = 'tbb' # requires tbb to be installed
    
    for folder in [f'{CWD}/models', f'{CWD}/figs/voxdyn']:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    parser = ArgumentParser()

    parser.add_argument("-n",  "--node2node",                default='hyper',   type=str, action="store", help="TrackML dataset re-sample: Ground truth adjacency matrix type")
    parser.add_argument("-m0", "--min_points_per_centroid",  default=1,         type=int, action="store", help="Minimum number of points per K-means cell assignment")
    parser.add_argument("-m1", "--max_points_per_centroid",  default=int(1E9),  type=int, action="store", help="Maximum number of points per K-means cell assignment")
    
    parser.add_argument("-e0", "--event_start", default=[1000, 7700],    type=int,      nargs='+',       help="Event file start")
    parser.add_argument("-e1", "--event_end",   default=[2399, 8699],    type=int,      nargs='+',       help="Event file end")
    parser.add_argument("-i",  "--niter",       default=20,              type=int,      action="store",  help="Number of iterations")
    parser.add_argument("-c",  "--ncell",       default=[65536],         type=int,      nargs='+',       help='Voxel count (e.g. --ncell 65536 131072) (use only one per run if RAM limited)')
    parser.add_argument("-w",  "--weighted",    default=0,               type=int,      action="store",  help='Use per hit weights')
    parser.add_argument("-b",  "--maxbuffer",   default=np.int64(2.1E9), type=np.int64, action="store",  help='Maximum buffer size (increase with large train samples)')
    parser.add_argument("-d",  "--device",      default='cpu',           type=str,      action="store",  help="Device (cpu, cuda)")
    
    args = parser.parse_args()
    
    if type(args.ncell) is int: # if only one given
        args.ncell = [args.ncell]
    
    print(args)
    
    event_range = iotools.construct_event_range(event_start=args.event_start, event_end=args.event_end)
    print(event_range)
    
    ## Train estimator
    voxdyn.train_voxdyn(event_loader=trackml.load_reduce_event,
                        event_range=event_range,
                        ncell_list=args.ncell,
                        niter=args.niter,
                        min_points_per_centroid=args.min_points_per_centroid,
                        max_points_per_centroid=args.max_points_per_centroid,
                        node2node=args.node2node,
                        maxbuffer=args.maxbuffer,
                        weighted=args.weighted,
                        device=args.device)

if __name__ == '__main__':
    main()
