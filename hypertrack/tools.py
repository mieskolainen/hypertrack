# Graphs and other processing tools
#
# m.mieskolainen@imperial.ac.uk, 2023

import numba
import pandas as pd
import numpy as np
import networkx as nx
import scipy
import random
from prettytable import PrettyTable
from functools import wraps
from time import time
from typing import Optional, List

from math import floor, ceil, log10

@numba.njit(fastmath=True)
def select_valid(x):
    mask = x != -1
    return x[mask]

def attach_fully_connected_node(A, d=1.0):
    """
    Takes in a sparse graph (csr_matrix), possibly containing
    multiple disconnected subgraphs and attachs a 
    node to the graph that is connected to all other nodes
    
    Args:
        A:  real valued adjacency matrix (csr_matrix)
        d:  value for the connection
    
    Returns:
        extended adjacency matrix
    """
    
    A2 = A.copy()
    A2 = scipy.sparse.vstack((A2, np.ones((1,A2.shape[1]))*d))
    A2 = scipy.sparse.hstack((A2, np.ones((A2.shape[0],1))*d))

    return A2.tocsr()

def running_mean_uniform_filter1d(x: np.ndarray, N: int):
    """
    Running mean filter
    
    Args:
        x: input array
        N: filter window size
    
    Returns:
        filtered array
    """
    N = min(len(x), N)
    df = pd.DataFrame({'B': x})    
    return df.rolling(window=N).mean().to_numpy()

def split(a: int, n: int):
    """
    Generator which returns approx equally sized chunks.
    Args:
        a : Total number
        n : Number of chunks
    Example:
        list(split(10, 3))
    """
    if len(a) < n: # Overflow protection
        n = len(a)
    
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def split_size(a: int, n: int):
    """
    As split_start_end() but returns only size per chunk
    """
    ll  = list(split(a,n))
    out = [len(ll[i]) for i in range(len(ll))]

    return out

def explicit_range(entry_start: int, entry_stop: int, num_entries: int):
    """
    Clean None from entry_start and entry_stop
    """
    start = 0 if entry_start is None else entry_start
    stop  = num_entries if entry_stop is None else entry_stop

    return start, stop

def split_start_end(a, n: int, end_plus:int=1):
    """
    Returns approx equally sized chunks.

    Args:
        a:        Range, define with range()
        n:        Number of chunks
        end_plus: Python/nympy index style (i.e. + 1 for the end)

    Examples:
        split_start_end(range(100), 3)  returns [[0, 34], [34, 67], [67, 100]]
        split_start_end(range(5,25), 3) returns [[5, 12], [12, 19], [19, 25]]
    """
    ll  = list(split(a,n))
    out = []

    for i in range(len(ll)):
        out.append([ll[i][0], ll[i][-1] + end_plus])

    return out

def timing(f):
    """
    Timing function wrapper
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        #print(f'func:{f.__name__} args:[{args}, {kw}] took: {te-ts:2.4f} sec')
        print(f'func:{f.__name__} took: {te-ts:2.4f} sec')
        return result
    return wrap

@numba.njit(fastmath=True)
def backtrack_linear_index(ind: List[bool], k: int):
    """
    Find the k-th index which is True
    
    (should optimize this function, i.e. without for loop)
    
    Args:
        ind : boolean list
        k   : index
    """
    s = -1
    for i in range(len(ind)):
        if ind[i]: # Is true
            s += 1
            if s == int(k): # Found it
                return i
    return -1

def normalize_cols(x: np.ndarray, order=2):
    """ Normalize matrix columns to the L-p norm """
    return x / np.linalg.norm(x, ord=order, axis=0, keepdims=True)

def normalize_rows(x: np.ndarray, order=2):
    """ Normalize matrix rows to the L-p norm """
    return x / np.linalg.norm(x, ord=order, axis=1, keepdims=True)

@numba.njit(fastmath=True)
def count_reco_cluster(x_ind: np.ndarray, hits_min: int=4):
    """
    Count the number of reconstructable clusters
    
    Args:
        x_ind:    hit indices per cluster array
        hits_min: minimum number of hits
    
    Returns:
        count
    """
    return np.sum(np.sum(x_ind != -1, 1) >= hits_min)

@numba.njit(parallel=True, fastmath=True)
def compute_hit_mixing(x_ind: np.ndarray, x_ind_hat: np.ndarray, match_index: np.ndarray, hits_max: int=30):
    """
    Compute (true,reco) hit multiplicity mixing matrix
    
    Args:
        x_ind:        ground truth hit indices per cluster array
        x_ind_hat:    estimated hit indices per cluster array
        match_index:  match pointing indices from estimate --> ground truth
        hits_max:     the maximum number of hits
    
    Returns:
        H:      hit count mixing 2D-square matrix array (true, reco)
        eff:    reconstruction hit set efficiency
        purity: reconstruction hit set purity
    """
    H          = np.zeros((hits_max, hits_max), dtype=np.float32)
    purity     = (-1)*np.ones(x_ind.shape[0], dtype=np.float32)
    eff        = (-1)*np.ones(x_ind.shape[0], dtype=np.float32)
    O          = x_ind_hat.shape[0]
    n_overflow = 0
    
    for i in numba.prange(O):
        if match_index[i] != -1:
            
            # True set
            x_ind__  = select_valid(x_ind[match_index[i],:])
            N_true   = len(x_ind__)
            true_set = set(x_ind__)
            
            # Reco set
            x_ind_hat__ = select_valid(x_ind_hat[i,:])
            N_reco      = len(x_ind_hat__)
            reco_set    = set(x_ind_hat__)
            
            # Intersect
            IS = true_set.intersection(reco_set)

            # Metrics
            purity[match_index[i]] = len(IS) / N_reco
            eff[match_index[i]]    = len(IS) / N_true
            
            if N_true >= hits_max or N_reco >= hits_max: # Overflow protection
                n_overflow += 1
                continue
            
            H[N_true, N_reco] += 1
    
    if n_overflow > 0:
        print(f"compute_hit_mixing: Histogram overflows {n_overflow}")
    
    return H, eff, purity

@numba.njit(parallel=True, fastmath=True)
def create_cluster_labels(N: int, x_ind: np.ndarray):
    """
    Create cluster labelings
    
    Args:
        N:              Number of nodes (hits)
        x_ind:          Hit indices of tracks list
        outlier_value:  Default cluster value for unassigned hits

    Returns:
        labels:         List of cluster label integers
    """

    # Assign -1 for the hits not assigned to any ground truth cluster
    labels = (-1) * np.ones(N, dtype=np.int64)

    # Loop over ground truth assignments
    O = x_ind.shape[0]
    for c in numba.prange(O):
        index = select_valid(x_ind[c,:])
        labels[index] = c
    
    return labels

@numba.njit(parallel=True, fastmath=True)
def create_cluster_ind(labels: np.ndarray, hmax: int =100):
    """
    Create cluster index arrays
    
    Args:
        labels: cluster labels list of integers
        hmax:   maximum number of hits per cluster
    Returns:
        cluster assignments [K x hmax]
    """
    K   = np.max(labels)
    if K == -1: K = 1 # No clusters special case
    
    out = (-1)*np.ones((K, hmax), dtype=np.int64) # Default value -1
    
    for n in numba.prange(K):
        ind    = np.where(labels == n)[0]
        maxv   = min(hmax, len(ind))
        out[n, 0:maxv] = ind[0:maxv]

    return out

@numba.njit(fastmath=True)
def count_hit_multiplicity(M: np.ndarray, bins:int =None):
    """
    Count the number of hits per cluster for diagnostics
    
    Args:
        M:        array of clusters (rows) and corresponding assingments (cols)
        bins:     number of histogram bins
    
    Returns:
        counts:   multiplicity histogram
        overflow: number of overflows
    """
    bins   = M.shape[1] if bins is None else bins
    counts = np.zeros(bins, dtype=np.int64)
    overflow = 0
    K = M.shape[0]

    for i in range(K):
        summ = np.sum(M[i,:] != -1)
        if summ < bins:
            counts[summ] += 1
        else:
            overflow += 1

    return counts, overflow

@numba.njit(parallel=True, fastmath=True)
def span_adj_mat(x_ind: np.ndarray, N: int, symmetric: bool=True):
    """
    Create boolean adjacency matrix from an index array
    such that the first element [0] is the pivot which spans
    the edges to other elements in the index array.
    
    Args:
        x_ind:     index array (K x max) with -1 for null elements
        N:         dimension of A
        symmetric: symmetric (undirected) adjacency

    Returns:
        adjacency matrix
    """
    A = np.zeros((N,N), dtype=np.bool_)
    rows,cols = x_ind.shape[0], x_ind.shape[1]
    
    for n in numba.prange(rows):
        a = x_ind[n,0]
        for j in range(cols):
            b = x_ind[n,j]
            if (a != -1) and (b != -1):
                A[a,b] = True
                if symmetric:
                    A[b,a] = True
    return A

@numba.njit(fastmath=True)
def adj2edges(A: np.ndarray):
    """
    Convert an adjacency matrix to an edge list

    Args:
        A:          adjacency matrix (N x N)
    
    Returns:
        edge_index: adjacency list (2 x E)
    """
    
    return np.vstack(A.nonzero()), A

@numba.njit(parallel=True, fastmath=True)
def edges2adj(edge_index: np.ndarray, N: int):
    """
    Convert an edge list to an adjacency matrix A
    
    Args:
        edge_index: adjacency list (2 x E)
        N:          dimension of A (N x N)
    
    Returns:
        A:          adjacency matrix
    """
    A = np.zeros((N,N), dtype=np.bool_)

    for n in numba.prange(edge_index.shape[1]):
        i = edge_index[0,n]
        j = edge_index[1,n]
        A[i,j] = True
    
    return A

@numba.njit(parallel=True, fastmath=True)
def create_edge_label(edge_index_hat: np.ndarray, A: np.ndarray):
    """
    Args:
        edge_index_hat: estimated adjacency list (2 x E)
        A:              Ground truth adjacency matrix (N x N)
    Returns:
        edge_label:     edge_label list (E)
    """
    E = edge_index_hat.shape[1]
    edge_label = np.zeros(E, dtype=np.bool_)

    for n in numba.prange(E):
        i = edge_index_hat[0,n] 
        j = edge_index_hat[1,n]
        edge_label[n] = A[i,j]
    
    return edge_label

def track_minimal_spanning_tree(X: np.ndarray, x_ind_single: np.ndarray):
    """
    Compute minimal spanning tree according to hit to hit distances in X
    
    Args:
        X:             hit (node) matrix array (N x dim)
        x_ind_single:  single track hit indices (array)

    Returns:
        minimal spanning tree adjacency with indices corresponding in x_ind_single
    """

    nhits = np.sum(x_ind_single != -1)

    if nhits == 0:
        raise Exception(__name__ + f'.track_minimal_spanning_tree: No hits (all == -1)')

    # Compute L2-distance matrix
    D = np.zeros((nhits, nhits))
    for i in range(nhits):
        a = x_ind_single[i]
        for j in range(nhits):
            b = x_ind_single[j]
            D[i,j] = np.linalg.norm(X[a,:] - X[b,:])

    # -----------------------------------------
    # Minimum spanning tree with networkx
    
    G = nx.cycle_graph(nhits)
    for i in range(nhits):
        for j in range(nhits):
            if i != j:
                G.add_edge(i, j, weight=D[i,j])

    T = nx.minimum_spanning_tree(G, algorithm='kruskal')
    mst_edges = sorted(T.edges(data=True))

    if len(mst_edges) != (nhits - 1): # hit - hit - ... - hit (number of edges == N - 1)
        print(mst_edges)
        raise Exception(f'track_minimal_spanning_tree: Spanning tree fails with the track (check the input)')

    return mst_edges

@numba.njit(fastmath=True)
def compute_hyper(edge_index: np.ndarray, x_ind: np.ndarray):
    """
    Helper function for compute_ground_truth()
    """
    z = 0
    
    # First process all particles which had hits
    for n in range(x_ind.shape[0]):
        nhits = len(select_valid(x_ind[n,:]))

        found_self = []
        for i in range(nhits):
            a = x_ind[n,i]
            for j in range(nhits):
                b = x_ind[n,j]

                # Self-edges only once
                if a == b:
                    if a not in found_self:    
                        edge_index[:,z] = a # edge a --> a
                        found_self.append(a)
                else:
                    edge_index[:,z] = np.array([a,b], dtype=np.int64) # edge a --> b
                
                z += 1
    return z

def compute_ground_truth_A(X: np.ndarray, x_ind: np.ndarray, node2node: str='hyper',
                           self_connect_noise: bool=False, edgeMAX: int=int(1e7)):
    """
    Compute Ground Truth adjacency structure
    
    Args:
        X:          input data
        x_ind:      hit indices per object (track)
        node2node:  ground truth construction type
                    'hyper':    all hits of the track connects with a 'lasso' (hyperedge)
                    'eom':      natural minimal EOM (eq.of.motion) trajectory (linear topology)
                    'cricket':  next to minimal i.e. EOM + 'double hops' included
        self_connect_noise:  connect noise hits with-self loops (True, False)
        edgeMAX:             number of edges to construct at maximum (keep it high enough)
    
    Returns:
        adjacency list (2 x N)
    """
    edge_index = (-1)*np.ones((2, edgeMAX), dtype=np.int64)
    z = 0

    # Construct adjacency according to the minimum spanning tree
    if   node2node == 'eom' or node2node == 'cricket':
            
        # First process all particles which had hits
        for n in range(x_ind.shape[0]):

            mst_edges = track_minimal_spanning_tree(X=X, x_ind_single=x_ind[n,:])
            #print(mst_edges)

            # We have only one hit (minimal spanning tree is empty)
            if mst_edges == []:
                a = x_ind[n,0]
                edge_index[:,z] = a # self-edge only
                z += 1

            # We have at least two hits
            else:
                found_self = []
                for i in range(len(mst_edges)):

                    a = x_ind[n, mst_edges[i][0]]
                    b = x_ind[n, mst_edges[i][1]]

                    edge_index[:,z] = np.array([a,b], dtype=np.int64); z += 1     # edge  a --> b
                    edge_index[:,z] = np.array([b,a], dtype=np.int64); z += 1     # edge  b --> a
                    
                    if a not in found_self:
                        edge_index[:,z] = a; z += 1 # self-edge, only once a --> a
                        found_self.append(a)

                    if b not in found_self:
                        edge_index[:,z] = b; z += 1 # self-edge, only once b --> b
                        found_self.append(b)

                # Now add the additional double hop connections
                if node2node == 'cricket':
                    for i in range(len(mst_edges)-1):
                        a = x_ind[n, mst_edges[i][0]]
                        b = x_ind[n, mst_edges[i+1][1]] # + 1
                        
                        edge_index[:,z] = np.array([a,b], dtype=np.int64); z += 1     # edge  a --> b
                        edge_index[:,z] = np.array([b,a], dtype=np.int64); z += 1     # edge  b --> a

    # Construct the full hit-to-every-other-hit of the track i.e. 'hyperedge'
    elif node2node == 'hyper':
        z = compute_hyper(edge_index=edge_index, x_ind=x_ind)
    else:
        raise Exception(__name__ + f'.compute_ground_truth_A: Unknown node2node mode chosen = {node2node}')

    # Finally, find missing indices (these are hits in X which are e.g. noise because not found in x_ind)
    all_ind = np.arange(0, len(X))
    found   = np.intersect1d(x_ind[x_ind != -1].flatten(), all_ind)
    noise   = np.setdiff1d(all_ind, found)

    # Add to adjacency
    if self_connect_noise:
        for i in range(len(noise)):
            edge_index[:,z] = noise[i]
            z += 1

    edge_index = edge_index[:, 0:z] # Truncate empty buffer
    
    print(__name__ + f'.compute_ground_truth_A: Edges = {edge_index.shape[1]} | Found {len(noise)} ({len(noise)/len(X):0.3E}) unassociated hits (self_connect_noise = {self_connect_noise})')

    return edge_index

def print_clusters(x_ind):
    """
    Print cluster information
    
    Args:
        x_ind:  cluster ground truth array
    """
    for i in range(x_ind.shape[0]):
        valid = select_valid(x_ind[i,:])
        print(f'[{i:4d}]: N = {len(valid):2d} {valid}')

class Graph:
    """
    A simple class for graphs
    """
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]
 
    def DFSUtil(self, temp, v, visited):
 
        # Mark the current vertex as visited
        visited[v] = True
 
        # Store the vertex to list
        temp.append(v)
 
        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:
 
                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp

    def addEdge(self, v, w):
        """ Add an undirected edge """
        self.adj[v].append(w)
        self.adj[w].append(v)
    
    def connectedComponents(self):
        """
        Retrieve connected components in an undirected graph
        """
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc

def find_connected_components(edge_index: np.ndarray, N: int):
    """
    Find connected components based on the adjacency list

    Args:
        edge_index: Adjacency list (2 x E)
        N:          Number of nodes in the graph
    
    Returns:
        connected components
    """
    # Create edgelist
    g = Graph(N)

    for n in range(edge_index.shape[1]):
        g.addEdge(edge_index[0,n], edge_index[1,n])

    return g.connectedComponents()

def closure_test_adj(edge_index: np.ndarray, x_ind: np.ndarray, N: int, DEBUG=False):
    """
    Unit test the constructed adjacency structures
    against a ground truth
    
    Args:
        edge_index:  Adjacency list (2 x E)
        x_ind:       Hit indices per tracks array
        N:           Number of hits
    """

    if N != np.max(x_ind.flatten())+1:
        raise Exception(f'closure_test_adj: N ({N}) != np.max(x_ind) + 1 ({np.max(x_ind.flatten())+1})')
    
    cc = find_connected_components(edge_index=edge_index, N=N)
    
    if DEBUG:
        print(cc)

    n_track_rec = len(cc)
    n_track_tru = len(x_ind)

    print("Closure unit test: (connected component) track candidates")
    print(f"number of track using true A adjacency = {n_track_rec} (edge_index)")
    print(f"number of true track                   = {n_track_tru} (x_ind)")

    return False if n_track_rec != n_track_tru else True

def value_and_uncertainty_scientific(x, dx, one_digit=True, rounding='round'):
    """
    A value and its uncertainty (or dispersion measure)
    in the scientific x(dx) format
    
    Args:
        x:         value
        dx:        dispersion
        one_digit: force only one digit output for dx
        rounding:  rounding of dx, 'ceil', 'floor' or 'round'
    
    Returns:
        a string with x(dx) format 
    """
    unc = dx / (10**floor(log10(dx)))
    
    if   rounding == 'ceil':
        unc = ceil(unc)
    elif rounding == 'floor':
        unc = floor(unc)
    elif rounding == 'round':
        unc = round(unc)
    else:
        raise Exception('value_and_uncertainty_scientific: Unknown rounding')
    
    prec = -floor(log10(dx))
    
    if one_digit and unc == 10:
        unc  = 1
        prec = prec - 1
    
    text = '{:.{prec}f}({:.0f})'.format(x, unc, prec=prec)
    
    print(f'{x:0.6f} {dx:0.6f} : {text}')
    
    return text
