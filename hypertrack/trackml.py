# TrackML kaggle challenge dataset processing functions
#
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np
import pickle
import pandas as pd
import numpy as np
import numba
import os
from termcolor import cprint
from typing import List

from hypertrack import tools, voxdyn

def process_data(PATH:str, maxhits_per_trk:int=30, verbose:bool=False):
    """
    Read and process TrackML kaggle challenge tracking dataset
    
    https://www.kaggle.com/competitions/trackml-particle-identification/data
    
    Args:
        path:              path to TrackML files
        maxhits_per_trk:   maximum number of hits per track
        verbose:           verbose output
    
    Returns:
        dictionary with keys
        X:                 every single hit (x, y, z, volume_id, layer_id, module_id)
        x_ind:             indices pointing to X for particles with at least 1 hit,
                           with running order matching particles[visible_particle]
        particles:         generated particles [particle id, vx,vy,vz, px,py,pz, q, nhits]
        visible_particle:  boolean array with length of particles (at least 1 hit == True, else False)
        hit_weights:       weight for every single hit
    """
    
    HITS_PATH      = PATH + '-hits.csv'
    CELLS_PATH     = PATH + '-cells.csv'
    TRUTH_PATH     = PATH + '-truth.csv'
    PARTICLES_PATH = PATH + '-particles.csv'

    hits      = pd.read_csv(HITS_PATH)
    cells     = pd.read_csv(CELLS_PATH)
    truth     = pd.read_csv(TRUTH_PATH)
    particles = pd.read_csv(PARTICLES_PATH)
    
    if verbose:
        print(f'hits: \n {hits}')
        print(f'cells: \n {cells}')
        print(f'truth: \n {truth}')
        print(f'particles: \n {particles}')        

    # Collect ALL hits (including also noise hits)
    X = np.array([hits['x'], hits['y'], hits['z'], hits['volume_id'], hits['layer_id'], hits['module_id']]).T
    hit_weights = np.array(truth['weight'])

    # Collections
    truth_particle_id     = np.array(truth['particle_id'])
    truth_hit_id          = np.array(truth['hit_id'])
    particles_particle_id = np.array(particles['particle_id'])

    # Find hit collections per particle which has at least 1 hit
    N_particles      = len(particles)
    visible_particle = np.zeros(N_particles, dtype=bool)
    x_ind = (-1)*np.ones((len(particles[particles['nhits'] > 0]), maxhits_per_trk), dtype=int)

    # Loop over all event (generated) particles
    # (Numba is slower for this loop)
    k = 0
    for i in range(len(particles_particle_id)):
        hit_index = truth_hit_id[truth_particle_id == particles_particle_id[i]]

        if len(hit_index) != 0:
            x_ind[k, 0:len(hit_index)] = hit_index - 1 # -1 because dataframe starts indexing from 1
            visible_particle[i] = True                 # Tag it containing at least 1 hit
            k += 1
    
    # Conversions
    X = X.astype(np.float32)
    X = np.ascontiguousarray(X) # make it C-contiguous
    
    hit_weights = hit_weights.astype(np.float32)
    particles   = particles.to_numpy().astype(np.float32)
    
    return {'X': X, 'x_ind': x_ind, 'particles': particles, 'visible_particle': visible_particle, 'hit_weights': hit_weights}


def load_reduce_event(event:int, rfactor:float=None, noise_ratio:float=None,
                      X_DIM:List[int]=[0,1,2], P_DIM:List[int]=[1,2,3,4,5,6,7,8]):
    """
    Load event and reduce pile-up
    
    Args:
        event:         event number index
        rfactor:       pile-up reduction factor
        noise_ratio:   noise fraction
        X_DIM:         cluster (hit info) parameter indices
        P_DIM:         cluster (latent info) parameter indices

    Returns:
        X:             hit (node) data
        x_ind:         cluster ground truth index array pointing to X
        objects:       cluster latent object data
        hit_weights:   hit (node) weights
    """
    CWD = os.getcwd()
    
    with open(f'{CWD}/data/trackml_event{event:09d}.pkl', 'rb') as handle:
        data_obj = pickle.load(handle)

    X               = data_obj['X'][:, X_DIM] # We pick here the 3D-coordinate information only
    x_ind           = data_obj['x_ind']
    objects         = data_obj['particles']
    visible_objects = data_obj['visible_particle']
    hit_weights     = data_obj['hit_weights']
    
    # Pick (visible) particle data (at least 1 hit containing)
    objects = objects[visible_objects]
    
    # Pick chosen latent information (vertex 3-position, 3-momentum ...)
    objects = objects[:, P_DIM]
    
    n_in    = x_ind.shape[0]
    
    if rfactor is not None and noise_ratio is not None:    
        
        # Reduce number of tracks
        X, x_ind, hit_weights, hot_ind = reduce_tracks( \
            X=X, x_ind=x_ind, hit_weights=hit_weights, rfactor=rfactor, noise_ratio=noise_ratio)
            
        # Now pick objects which survived the reduction
        objects = objects[hot_ind]
    
    n_out = x_ind.shape[0]

    cprint(__name__ + f'.load_reduce_event: Input (output) tracks = {n_in} ({n_out}) [{n_out/n_in:0.03f}]', 'yellow')
    
    return X, x_ind, objects, hit_weights


@tools.timing
def generate_graph_data(event:int, rfactor:float, noise_ratio:float, node2node:str, geom_param:dict, geometric_obj:dict):
    """
    Generate TrackML data graph input (both training and inference data)

    Args:
        event:       event file number
        rfactor:     pile-up reduction factor
        node2node:   ground truth adjacency construction mode
        geom_param:  geometric pre-algorithm algorithm parameters
        voronoi_obj: pre-loaded voronoi estimator objects

    Returns:
        dictionary with objects
    """
    
    # ------------------------------------------------------------------------
    # TrackML data

    # Read event
    X, x_ind, objects, hit_weights = load_reduce_event(event=event, rfactor=rfactor, noise_ratio=noise_ratio)

    # ------------------------------------------------------------------------
    # Count number of hits for diagnostics
    if geom_param['verbose']:
        counts, overflow = tools.count_hit_multiplicity(M=x_ind)

        print('Number of hits per track:')
        for i in range(len(counts)):
            print(f'[{i:2d}]: {counts[i]:5d}, fraction = {counts[i]/x_ind.shape[0] * 100:2.2f} %')    
    
    # ------------------------------------------------------------------------
    # 1. Compute the Ground Truth adjacency
    
    # N.B. self_connect_noise=False and noise hits are considered background (label=0)
    
    edge_index = tools.compute_ground_truth_A(X=X, x_ind=x_ind, node2node=node2node, self_connect_noise=False)
    A = tools.edges2adj(edge_index=edge_index, N=X.shape[0])
    
    # ------------------------------------------------------------------------
    # 2. Run geometrical input adjacency predictor

    if   geom_param['algorithm'] == 'voxdyn': 
        edge_index_hat, A_hat, time_voxdyn   = voxdyn.voxdyn_predictor(X=X, obj=geometric_obj, device=geom_param['device'], node2node=node2node)
    elif geom_param['algorithm'] == 'neurodyn':
        raise Exception('Neurodyn not implemented.')
        #edge_index_hat, A_hat, time_neurodyn = neurodyn.neural_predictor(X=X, obj=geometric_obj, **geom_param)
    else:
        raise Exception('Unknown geometrical algorithm chosen.')
    
    # ------------------------------------------------------------------------
    # 3. Create ground truth labels, but restricted to our A_hat adjacency!
    # That is the graph what the network will see also in the deployment (inference) phase
    
    edge_label = tools.create_edge_label(edge_index_hat=edge_index_hat, A=A)
    
    if geom_param['verbose']:
        tools.print_graph_metrics(A=A, A_hat=A_hat)

    d = {}
    for variable in ['X', 'A', 'A_hat', 'x_ind', 'edge_index', 'edge_index_hat', 'edge_label', 'objects', 'hit_weights']:
        d[variable] = eval(variable)

    return d

@numba.njit(fastmath=True)
def reduce_tracks(X: np.ndarray, x_ind: np.ndarray, hit_weights:np.array, rfactor:float, noise_ratio:float=-1):
    """
    Reduce tracks (pileup) of the event by a factor (0,1]
    
    The reduction is simplistic 'overall' reduction of tracks, good enough
    for benchmarking purposes.
    
    (even more realistically one should drop the number of proton-proton vertices and the associated particles & hits)
    
    Args:
        X:               node data (N x dim)
        x_ind:           a list of list of indices in X per track (track) object
        hit_weights:     weights per hit (N)
        rfactor:         reduction factor (0,1]
        noise_ratio:     noise hit ratio (0,1] (set -1 for the original, set None for no noise)
    
    Returns:
        X_new:           as above in the input
        x_ind_new:       as above in the input
        hit_weights_new: as above in the input
        hot_ind:         chosen subset row indices of x_ind (input)
    """
    
    n_tracks_all = x_ind.shape[0]

    if rfactor < 1.0:
        # Reduce and Poisson fluctuate
        n_tracks = max(1, int(rfactor * n_tracks_all))
        n_tracks = min(max(1, np.random.poisson(lam=n_tracks)), n_tracks_all)
    else:
        n_tracks = n_tracks_all
    
    # Create unique random subset of particles (no repetitions)
    hot_ind = np.random.choice(np.arange(0, n_tracks_all), size=n_tracks, replace=False)

    # Construct new indices
    x_ind_new = (-1)*np.ones((n_tracks, x_ind.shape[1]), dtype=np.int64)
    
    z = 0 # running new index
    for n in range(n_tracks):
        for i in range(x_ind.shape[1]):
            if x_ind[hot_ind[n],i] != -1:
                x_ind_new[n,i] = z
                z += 1
            else:
                break # Break inner loop
    
    # Construct new X
    X_new = np.zeros((z, X.shape[1]), np.float32)
    hit_weights_new = np.zeros(z, dtype=np.float32)
    
    z = 0 # Start from 0
    for n in range(n_tracks):
        for i in range(x_ind.shape[1]):
            k = x_ind[hot_ind[n],i] # pick single hit index
            if k != -1:
                X_new[z,:]         = X[k,:]
                hit_weights_new[z] = hit_weights[k] 
                z += 1
            else:
                break # Break inner loop
    
    # -------------------------------------------
    if noise_ratio is not None:
        
        # Add unassociated hits (noise) hits as in the original data
        flat        = x_ind.flatten()
        hit_ind     = np.unique(flat[flat != -1])
        noise_ind   = np.array(list(set(np.arange(0,len(X))) - set(hit_ind)), dtype=np.int64) # set difference

        if noise_ratio < -0.0: # Use the original input data ratio
            noise_ratio = len(noise_ind) / len(X)
        
        # Compute noise count and forbid overshoot, cannot have more noise hits than there exists in original
        N_noise  = int(np.random.poisson(lam = len(X_new) * noise_ratio / (1 - noise_ratio)))
        N_noise  = min(len(noise_ind), N_noise)
        pick_ind = np.random.choice(a=noise_ind, size=N_noise, replace=False)
        
        # Add them last
        X_new           = np.concatenate((X_new, X[pick_ind,:]), axis=0)
        hit_weights_new = np.concatenate((hit_weights_new, hit_weights[pick_ind]), axis=0)
    
    # Change format
    X_new = X_new.astype(np.float32)
    X_new = np.ascontiguousarray(X_new) # C-contiguous

    # Normalize the sum over the new hit weights to sum to 1
    hit_weights_new /= np.sum(hit_weights_new)
    
    return X_new, x_ind_new, hit_weights_new, hot_ind
