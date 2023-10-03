# Voxel-Dynamics graph adjacency predictor
#
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np
import numba
import faiss
import time
import pickle
import csr
import gc
import os
import scipy

from numba.typed import List
from termcolor import cprint

from hypertrack import tools, iotools, visualize
from hypertrack.tools import select_valid

@tools.timing
@numba.njit(parallel=True, fastmath=True)
def train_2pt_connectivity(C: np.ndarray, linear_list: List[List[int]], I: np.ndarray, A_track: List[np.ndarray]):
    """
    Train a 2-point connectivity C-matrix

    Args:
        C:           a cell 2-point connectivity matrix (V x V) (csr-library matrix)
        linear_list: a list of lists
        I:           index assignments
        A_track:     ground truth 'micro adjacency' per track, a list of adjacency matrices
        ncell:       number of V-cells
    
    Returns:
        Values are filled in the input C
    """
    
    L = len(linear_list)
    
    for p in numba.prange(L):
        
        # Find in which cell hits this track belongs to
        cell_ind = I[linear_list[p]]
        N        = len(cell_ind)
        
        for i in range(N):
            u = cell_ind[i][0] # [0] removes outer []
            for j in range(N):
                v = cell_ind[j][0]  # [0] removes outer []
                
                # Check ground truth adjacency
                if A_track[p][i,j]:
                    C[u,v] = True

@tools.timing
@numba.njit(parallel=True, fastmath=True)
def train_2pt_connectivity_sparse(maxbuffer: np.int64, linear_list: List[List[int]], I: np.ndarray, A_track: List[np.ndarray]):
    """
    Train a 2-point connectivity matrix [sparse matrix version]

    Args:
        maxbuffer:   maximum number of entry fills (int64)
        linear_list: a list of lists
        I:           index assignments
        A_track:     ground truth 'micro adjacency' per track, a list of adjacency matrices
        ncell:       number of V-cells
    
    Returns:
        row, col:    index arrays
    """
    row   = np.zeros(maxbuffer, dtype=np.int32) # Keep it 32 bits (number of V-cells is much less)
    col   = np.zeros(maxbuffer, dtype=np.int32)
    
    L     = len(linear_list)
    index = 0
    
    for p in range(L): # ! Do not use numba.prange here, because of running index !
        
        # Find in which cell hits this track belongs to
        cell_ind = I[linear_list[p]]
        N        = len(cell_ind)
        
        for i in range(N):
            u = cell_ind[i][0] # [0] removes outer []
            for j in range(N):
                v = cell_ind[j][0] # [0] removes outer []
                
                # Check ground truth adjacency
                if A_track[p][i,j]:
                    row[index] = int(u)
                    col[index] = int(v)
                    index   += 1

                    if index == maxbuffer:
                        print("train_2_pt_connectivity_sparse: Warning, maximum buffer size (" + str(maxbuffer) + ") reached.")
                        return row, col
    
    return row[0:index], col[0:index] # Note that the original memory allocation is leaked (gc.collect cannot free it)


@tools.timing
@numba.njit(parallel=True, fastmath=True)
def train_3pt_connectivity(C: np.ndarray, linear_list: List[List[int]], I: np.ndarray, A_track: List[np.ndarray]):
    """
    Train a 3-point connectivity tensor [EXPERIMENTAL]
    
    Args:
        C:           3-point connectivity tensor (V x V x V)
        linear_list: a list of lists
        I:           index assignments
        A_track:     ground truth 'micro adjacency' per track, a list of adjacency matrices
        ncell:       number of V-cells
    
    Returns:
        Values are filled in the input C
    """
    
    L = len(linear_list)
    
    for p in numba.prange(L):
        
        # Find in which cell hits this track belongs to
        cell_ind = I[linear_list[p]]
        N        = len(cell_ind)
        
        for i in range(N):
            u = cell_ind[i][0] # [0] removes outer []
            for j in range(N):
                v = cell_ind[j][0]  # [0] removes outer []
                for k in range(N):
                    w = cell_ind[k][0]  # [0] removes outer []

                    # Check ground truth adjacency
                    if A_track[p][i,j] and A_track[p][j,k] and A_track[p][k,i]:
                        C[u,v,w] = True


@numba.njit(parallel=True, fastmath=True)
def compute_A_direct(I: np.ndarray, C: np.ndarray):
    """
    "Direct" -- Construct hit adjacency matrix given the hit indices
    I which point which cell hits belong to
    
    Args:
        I:  cell indices for each hit (N)
        C:  cell 2-point connectivity matrix (V x V) (csr-library matrix)
    
    Returns:
        edge_index: adjacency list (2 x E)
        A:          adjacency matrix (N x N)
    """
    N = len(I)
    A = np.zeros((N,N), dtype=np.bool_) # (think about sparse matrices)
    
    for i in numba.prange(N):
        u     = I[i]
        C_row = C.row(u) # !csr library, get dense row        
        mask  = C_row[I]
        A[i, mask] = True
    
    return np.vstack(A.nonzero()), A


@numba.njit(parallel=True, fastmath=True)
def compute_A_reverse(I_per_V: np.ndarray, H: np.ndarray, C: np.ndarray, N: int):
    """
    "Reverse" -- Construct hit adjacency matrix given the hit indices
    in I_per_V which point which V-cell hits belong to
    
    Args:
        I_per_V:    cell indices for each hit (V x maxhits (buffer) per cell)
        H:          number of hits per V-space cell
        C:          cell connectivity matrix (V x V) (csr-library matrix)
        N:          adjacency matrix dimension
    
    Returns:
        edge_index: adjacency list (2 x E)
        A:          adjacency matrix (N x N)
    """
    
    A = np.zeros((N, N), dtype=np.bool_) # (think about sparse matrices)
    
    # Which V-cells have associated hits
    V_non_empty = np.where(H != 0)[0]
    V           = len(V_non_empty)
    
    # Loop over V-cell space
    for a in numba.prange(V):
        u = V_non_empty[a]
        
        C_row = C.row(u) # !csr library, get dense row
        
        for b in range(V):
            v = V_non_empty[b]
            
            # Can be connected according to the C-matrix
            if C_row[v]: # !csr library
                
                # Loop over associated real-space hits
                for i in I_per_V[u, 0:H[u]]:
                    
                    j      = I_per_V[v, 0:H[v]]
                    A[i,j] = True
    
    return np.vstack(A.nonzero()), A


@numba.njit(parallel=True, fastmath=True)
def compute_A_reverse_3pt(I_per_V3: np.ndarray, H3: np.ndarray, C3: np.ndarray, N: int):
    """
    3-pt version [EXPERIMENTAL]
    
    Args:
        I_per_V3:  cell indices for each hit (dim |V| x maxhits (buffer) per cell)
        H3:        number of hits per V-space cell
        C3:        cell connectivity matrix (dim |V| x dim |V| x dim |V|)
        N:         adjacency matrix dimension
    Returns:
       edge_index: adjacency list (2 x E)
       A:          adjacency matrix (N x N)
     
    """
    
    A1 = np.zeros((N, N), dtype=np.bool_) # (think about sparse matrices)
    A2 = np.zeros((N, N), dtype=np.bool_)
    A3 = np.zeros((N, N), dtype=np.bool_)
    
    # Which V-cells have associated hits
    V_non_empty = np.where(H3 != 0)[0]
    V = len(V_non_empty)
    
    # Loop over V-cell space
    for a in numba.prange(V):
        i = V_non_empty[a]
        
        for b in range(V):
            j = V_non_empty[b]
            
            for c in range(V):
                k = V_non_empty[c]

                if C3[i,j,k]: # Can be connected according to the C-matrix
                    
                    # Loop over associated real-space hits
                    for m in I_per_V3[i, 0:H3[i]]:
                        n       = I_per_V3[j, 0:H3[j]]
                        A1[m,n] = True
                    
                    for m in I_per_V3[j, 0:H3[j]]:
                        n       = I_per_V3[k, 0:H3[k]]
                        A2[m,n] = True
                    
                    for m in I_per_V3[k, 0:H3[k]]:
                        n       = I_per_V3[i, 0:H3[i]]
                        A3[m,n] = True
    
    A = A1 & A2 & A3
    
    return np.vstack(A.nonzero()), A


@numba.njit(parallel=True, fastmath=True)
def compute_I_per_V(I: np.ndarray, ncell: int, buffer: int):
    """
    From an event hit list to V-space cells
    
    Args:
        I:       index list of hits which point to V-space cell
        ncell:   number of V-cells
        buffer:  buffer length (make large enough!)
    
    Returns:
        I_per_V: hit index per V-cell
        (several hits possible, empty slots are -1)
        H:       number of hits per V-cell
    """
    I_per_V = (-1) * np.ones((ncell, buffer), dtype=np.int32)
    H = np.zeros(ncell, dtype=np.int32)
    N = len(I)

    for hit_index in numba.prange(N):
        v_index = I[hit_index]
        j = H[v_index]
        if j < buffer: # Cannot fill more than the buffer size
            I_per_V[v_index,j] = hit_index
            H[v_index] += 1

    return I_per_V, H


def load_particle_data(event:int, event_loader, node2node:str, weighted:bool=False):
    """
    Load particle data in a format suitable for training C-matrices (tensors)
    
    Args:
        event_loader:  event loader function handle
        event:         event index
        node2node:     target graph topology type
        weighted:      pick object weights
    
    Returns:
        X:        event data vectors
        A_track:  event adjacency matrix
        weights:  weights for each data vector
    """
    
    data_x        = None
    data_weights  = None
    A_track       = List()

    print(f'Event: {event}')

    # --------------------------------------------------------------------
    # Load event
    X, x_ind, objects, hit_weights = event_loader(event=event)
    # --------------------------------------------------------------------
    
    # Create ground truth adjacency matrix
    edge_index = tools.compute_ground_truth_A(X=X, x_ind=x_ind, node2node=node2node)
    A = tools.edges2adj(edge_index=edge_index, N=len(X))
    
    print(f'X.shape           = {X.shape}')
    print(f'x_ind.shape       = {x_ind.shape}')
    print(f'edge_index.shape  = {edge_index.shape} (min ind = {np.min(edge_index.flatten())} | max ind = {np.max(edge_index.flatten())})')
    print(f'objects.shape     = {objects.shape}')
    print(f'hit_weights.shape = {hit_weights.shape}')
    
    # --------------------------------------------------------------------
    # Pick visible objects mapped in x_ind
    index_list = []
    for i in np.arange(0, len(x_ind)):
        ind = select_valid(x_ind[i,:])
        A_track.append(A[np.ix_(ind, ind)])
        index_list.append(ind)
    
    # Unravel to one array
    choose_ind = np.concatenate(index_list).ravel()

    # Add to the full batch over different events
    data_x = X[choose_ind]
    if weighted:
        data_weights = hit_weights[choose_ind]

    return {'X': data_x, 'A': A_track, 'weights': data_weights}

@tools.timing
def combine_data(data: dict, weighted: bool):
    """
    Combine data from different events
    
    Args:
        data:         event-by-event dictionary data in a list
        weighted:     use weights
    
    Returns:
        data_x:       data vectors
        data_A:       adjacency per particle
        data_weights: weights
        linear_list:  indices of hits
    """
    linear_list  = List()
    k            = 0
    
    data_x       = None
    data_A       = List()
    data_weights = None

    # Concatenate data over all events
    for ev in range(len(data)):
        
        X       = data[ev]['X']
        A_track = data[ev]['A']
        weights = data[ev]['weights']

        # Over all tracks
        for i in range(len(A_track)):
            
            # Add track adjacency
            data_A.append(A_track[i].copy())

            # Add indices
            N  = len(A_track[i])
            linear_list.append(np.arange(k, k+N)) # This will contain int64 values
            k += N

        if ev == 0:
            data_x = X.copy()
            if weighted:
                data_weights = weights.copy()       
        else:
            data_x = np.concatenate((data_x, X.copy()), axis=0)
            if weighted:
                data_weights = np.concatenate((data_weights, weights.copy()), axis=0)

    return data_x, data_weights, data_A, linear_list

def faiss_to_gpu(index, device):
    """
    Faiss index object to GPU

    Args:
        index:  Faiss index object
        device: 'cpu' or 'cuda'
    
    Returns:
        index:  Index object
    """
    #if device != 'cpu' and torch.cuda.is_available():
    if device != 'cpu':    
        res   = faiss.StandardGpuResources()  # use a single GPU
        index = faiss.index_cpu_to_gpu(res, 0, index)
    return index

def train_voxdyn(event_loader, event_range: List[int], ncell_list:List[int]=[65536, 131072], ncell3:int=1024,
                 niter:int=50, min_points_per_centroid:int=1, max_points_per_centroid:int=int(1E9),
                 node2node:str='hyper', maxbuffer=np.int64(5E9), weighted=False, device:str='cpu', verbose:bool=True):
    """
    Train the geometric Voxel-Dynamics estimator
    
    Args:
        event_loader:               event loader function handle
        event_range:                event index list
        ncell_list:                 voxel counts to construct (list)
        ncell3:                     3-point voxel count (int)
        niter:                      number of K-means iterations
        min_points_per_centroid:    minimum number of points per centroid in K-means
        max_points_per_centroid:    maximum number of points per centroid in K-means
        node2node:                  target graph topology type
        maxbuffer:                  maximum (64 bit integer) number of entry fills for the sparse C-matrix
        weighted:                   use weights
        device:                     'cpu' or 'cuda'
        verbose:                    verbose print
        
    Returns:
        trained models are saved to the disk
    """
    
    # This should be good for Faiss
    faiss.omp_set_num_threads(os.cpu_count() // 2)
    
    # Load per event data
    data = [None] * len(event_range)
    for i in range(len(event_range)):
        data[i] = load_particle_data(event_range[i], event_loader, node2node, weighted)
    
    # Combine all event data
    data_x, hit_weights, A_track, linear_list = combine_data(data=data, weighted=weighted)
    
    # Free memory !
    del data
    gc.collect()
    
    # Make it C-contiguous for Faiss
    data_x = data_x.astype(np.float32)
    data_x = np.ascontiguousarray(data_x)

    if weighted:
        hit_weights = hit_weights.astype(np.float32)
        hit_weights = np.ascontiguousarray(hit_weights)
    
    # Produce statistics
    print('')
    print(f'- Total number of hits: {len(data_x)}')
    for i in range(data_x.shape[1]):
        print(f'- coord[{i}] (min,max,mean,std): {np.min(data_x[:,i])} | {np.max(data_x[:,i])} | {np.mean(data_x[:,i])} | {np.std(data_x[:,i])}')
    print('')
    
    dim   = data_x.shape[1]
    nhits = len(data_x)
    
    # 3-Point K-means [EXPERIMENTAL]
    """
    kmeans3 = faiss.Kmeans(d=dim, k=ncell3, niter=niter,
        min_points_per_centroid=min_points_per_centroid, 
        max_points_per_centroid=max_points_per_centroid,
        gpu=True if device != 'cpu' else False, verbose=verbose)

    if weighted:
        kmeans3.train(x=data_x, weights=hit_weights)
    else:
        kmeans3.train(x=data_x)
    
    centroids3 = kmeans3.centroids
    _, I3  = kmeans3.index.search(data_x, 1) # Find corresponding V-space position
    
    # Construct 3-point connectivity matrix
    
    print('Constructing C3-matrix ...')
    C3 = np.zeros((ncell3, ncell3), dtype=np.bool_)
    train_3pt_connectivity(C3=C3, linear_list=linear_list, I=I3, A_track=A_track)
    
    del I3
    del kmeans3
    gc.collect()
    """        
    
    centroids3 = None
    C3         = None
    
    for ncell in ncell_list:
        
        if device != 'cpu':
            iotools.showmem_cuda(device=device)
        else:
            iotools.showmem()
        
        # --------------------------------------------------------------------
        # Run Faiss K-means
        # min(max)_points_per_centroid is important, otherwise subsampling can occur!
        # https://github.com/facebookresearch/faiss/wiki/FAQ
        
        kmeans = faiss.Kmeans(d=dim, k=ncell, niter=niter,
            min_points_per_centroid=min_points_per_centroid, 
            max_points_per_centroid=max_points_per_centroid,
            gpu=True if device != 'cpu' else False, verbose=verbose)

        if weighted:
            kmeans.train(x=data_x, weights=hit_weights) # Weighted can fail with large number of clusters
        else:
            kmeans.train(x=data_x)
        
        centroids = kmeans.centroids
        _, I  = kmeans.index.search(data_x, 1) # Find corresponding V-space position
        
        del kmeans
        gc.collect()
        
        # --------------------------------------------------------------------
        # Clear memory if only one under processing
        if len(ncell_list) == 1:
            del data_x
            del hit_weights
            gc.collect()
        
        if device != 'cpu':
            iotools.showmem_cuda(device=device)
        else:
            iotools.showmem()
        
        # --------------------------------------------------------------------
        # Construct 2-point connectivity C-matrix
        
        print('Training 2-pt connectivity ...')
        
        ## Dense version
        #C_dense = np.zeros((ncell, ncell), dtype=np.bool_)
        #train_2pt_connectivity(C=C_dense, linear_list=linear_list, I=I, A_track=A_track)
        
        # Sparse version
        row, col = train_2pt_connectivity_sparse(maxbuffer=maxbuffer, linear_list=linear_list, I=I, A_track=A_track)
        
        print(__name__ + f'.train_voxdyn: Sparse C-matrix fills {len(row) / float(maxbuffer):0.3e} of maxbuffer ({maxbuffer:0.1e}) size')

        if device != 'cpu':
            iotools.showmem_cuda(device=device)
        else:
            iotools.showmem()        
        
        # --------------------------------------------------------------------
        # Clear memory if only one under processing
        if len(ncell_list) == 1:
            del linear_list
            del A_track
            del I
            gc.collect()
        
        if device != 'cpu':
            iotools.showmem_cuda(device=device)
        else:
            iotools.showmem()
        # --------------------------------------------------------------------
        
        ## Dense format into to sparse matrix
        #C  = scipy.sparse.csr_matrix(C_dense, dtype=np.bool_)
        #C3 = scipy.sparse.csr_matrix(C3_dense, dtype=np.bool_)
        
        # Row-col format into sparse matrix
        data = np.ones(len(row), dtype=np.bool_)
        C    = scipy.sparse.csr_matrix((data, (row, col)), shape=(ncell, ncell), dtype=np.bool_)
        C.sum_duplicates()
        C.sort_indices()
        
        ## Save to the disk
        CWD = os.getcwd()
        filename = f"{CWD}/models/voxdyn_node2node_{node2node}_ncell_{ncell}.pkl"
        with open(filename, "wb") as output_file:
            
            print(f'Saving voxdyn model information to: {filename}')
            data = {'ncell': ncell, 'ncell3': ncell3, 'centroids': centroids, 'centroids3': centroids3,
                    'C': C, 'C3': C3, 'node2node': node2node, 'nhits': nhits}
            pickle.dump(data, output_file, pickle.HIGHEST_PROTOCOL)
        
        ## Visualize
        if ncell <= 20000:
            visualize.plot_voronoi(centroids=centroids, savename=f'ncell_{ncell}')
            visualize.plot_C_matrix(C=C.toarray(), title=f'"{node2node}"', savename=f'ncell_{ncell}_node2node_{node2node}')
        
        if len(ncell_list) > 1:
            del centroids
            del I
            del C
            del row
            del col
            del data
            gc.collect()


def voxdyn_predictor(X:np.array, node2node:str='hyper', ncell:int=65536, obj=None,
                     adj_algorithm='reverse', buffer=15, device:str='cpu'):
    """
    Predict connectivity based on Voxel-Dynamic estimator

    Args:
        X:              input data array (C-contiguous, float32)
        node2node:      adjacency connectivity type
        ncell:          number of cells
        obj:            pre-loaded estimator object dictionary
        adj_algorithm:  'direct' or 'reverse' [identical result from both but latency may differ]
        buffer:         Node count default buffer size per V-cell for the 'reverse' algorithm
                        (will be recursively increased by x 10 until no saturation)              
        device:         computing device ('cpu' or 'cuda')
    
    Returns:
        edge_index_hat: adjacency list
        A_hat:          adjacency matrix
    """
    
    # This should be good for Faiss
    faiss.omp_set_num_threads(os.cpu_count() // 2)
    
    # Make sure it is float32 and C-contiguous (faiss requirements)
    X = X.astype(np.float32)
    X = np.ascontiguousarray(X)
    
    if obj is None:
        CWD = os.getcwd()
        try:
            with open(f"{CWD}/models/voxdyn_node2node_{node2node}_ncell_{ncell}.pkl", "rb") as input_file:
                obj = pickle.load(input_file)
        except Exception as e:
            cprint(e, 'red')
    
    cprint(__name__ + f".voxdyn_predictor: Using a model with ncell = {obj['ncell']} (2-pt) | node2node = {obj['node2node']}", 'green')

    # Inverted File Index with product quantization [EXPERIMENTAL]
    """
    nlist = 1000
    m     = 3
    nbits = 8
    quantizer = faiss_to_gpu(faiss.IndexFlatL2(dim), device)
    quantizer = faiss_to_gpu(faiss.IndexHNSWFlat(dim, m), device)
    index = faiss_to_gpu(faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits))
    index.train(obj['centroids'])
    index.add(obj['centroids'])
    """

    # Inverted File Index [EXPERIMENTAL]
    """
    nlist = 1000
    quantizer = faiss_to_gpu(faiss.IndexFlatL2(dim), device)
    index = faiss_to_gpu(faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2), device)
    index.add(obj['centroids'])
    """
    
    dim = X.shape[1]
    
    # --------------------------------------
    ## 2-point discretization search
    
    # 0. Span exact L2-metric search (this could be done only once outside this function, but this is fast)
    time_L2 = time.time()
    index = faiss_to_gpu(faiss.IndexFlatL2(dim), device)
    index.add(obj['centroids'])
    
    ## 1. Do the index search
    _, I_full = index.search(X, 1)
    index.reset() # Free Faiss memory
    I_full = I_full.squeeze()
    del index
    time_L2 = time.time() - time_L2

    # --------------------------------------
    ## 3-point discretization search (experimental)
    """
    index3 = faiss_to_gpu(faiss.IndexFlatL2(dim), device)
    index3.add(obj['centroids3'])
    
    D3_full, I3_full = index3.search(X, 1)
    index3.reset() # Free Faiss memory
    I3_full = I3_full.squeeze()
    del index3
    """
    # --------------------------------------
    
    # --------------------------------------
    ## Construct adjacency
    
    time_A  = time.time()
    
    if   adj_algorithm == 'direct':  # Direct algo
        
        edge_index_hat, A_hat = compute_A_direct(I=I_full, C=obj['C'])
    
    elif adj_algorithm == 'reverse': # Reverse algo
        
        # ---------
        # 2-pt version
        
        while True:
            
            I_per_V, H = compute_I_per_V(I=I_full, ncell=obj['ncell'], buffer=buffer)
            
            # Check that not no cell saturated the buffer
            max_occupancy = np.max(H)
            print(__name__ + f'.voxdyn_predictor: Max index occupancy per cell: {max_occupancy} | Buffer size = {buffer}')
            
            if max_occupancy == buffer:
                print(__name__ + f'.voxdyn_predictor: Increasing buffer size x 10 (set it higher manually)')
                buffer = buffer * 10
            else:
                break
        
        edge_index_hat, A_hat = compute_A_reverse(I_per_V=I_per_V, H=H, C=obj['C'], N=len(X))
        
        """
        # ---------
        # 3-pt version [EXPERIMENTAL]
        
        while True:
            
            I_per_V3, H3 = compute_I_per_V(I=I3_full, ncell=obj['ncell3'], buffer=buffer)
            
            # Check that not no cell saturated the buffer
            max_occupancy = np.max(H3)
            print(__name__ + f'.voxdyn_predictor: Max index occupancy per cell3: {max_occupancy} | Buffer size = {buffer}')

            if max_occupancy == buffer:
                print(__name__ + f'.voxdyn_predictor: Increasing buffer size x 10 (set it higher manually)')
                buffer = buffer * 10
            else:
                break
        
        edge_index_hat, A_hat = compute_A_reverse_3pt(I_per_V3=I_per_V3, H3=H3, C3=obj['C3'], N=len(X))
        """
    
    time_A = time.time() - time_A
    
    print(__name__ + f'.voxdyn_predictor: Edges = {np.sum(A_hat):0.1e}, Avg. degree {np.mean(np.sum(A_hat,1)):0.1f} | Geometric L2 search ({time_L2:0.4} s) | C-matrix look-up ({time_A:0.4} s)')
    
    return edge_index_hat, A_hat, {'time_L2': time_L2, 'time_A': time_A}
