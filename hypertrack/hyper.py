# HyperTrack neural clustering model core-logic functions
# 
# m.mieskolainen@imperial.ac.uk, 2023

import networkx as nx
import torch
import torch_geometric
import numpy as np
import time
import gc
from termcolor import cprint
from typing import Tuple, List

#import cugraph
import hdbscan
import sklearn

from hypertrack import hyperaux, tools, torchtools, losses, iotools

def hypertrack(model: dict, X: torch.Tensor, edge_index_hat: torch.Tensor,
            x_ind: np.ndarray=None, hit_weights: torch.Tensor=None,
            edge_label:torch.Tensor=None, true_clusters:torch.Tensor=None,
            objects:torch.Tensor=None, do_pdf:bool=False, do_clustering:bool=True,
            trainmode:bool=False, cluster_param:dict=None, train_param:dict=None, device:torch.device=torch.device('cuda'),
            DEBUG:bool=False):
    """
    HyperTrack model predictor function with embedded training losses
    
    Args:
        model:              neural model dictionary
        X:                  input graph node data
        edge_index_hat:     input graph adjacency
        x_ind:              hit to object associations array (training only)
        hit_weights:        ground truth node weights (training only)
        edge_label:         edge truth array (training only)
        true_clusters:      ground truth cluster associations per node (hit) (training only)
        objects:            ground truth object vector data (training only)
        do_pdf:             run pdf module
        do_clustering:      run clustering module
        trainmode:          training mode turned on == True
        cluster_param:      clustering parameters dictionary
        train_param:        training parameters dictionary
        device:             torch computing device
    
    Returns:
        cluster_labels_hat: list of clusters
        loss:               torch loss data
        aux:                auxialary information
    """
    
    loss = {
        'track_neglogpdf':     torch.tensor(0.0, device=device, dtype=X.dtype),
        'edge_BCE':            torch.tensor(0.0, device=device, dtype=X.dtype),
        'edge_contrastive':    torch.tensor(0.0, device=device, dtype=X.dtype),
        'cluster_BCE':         torch.tensor(0.0, device=device, dtype=X.dtype),
        'cluster_contrastive': torch.tensor(0.0, device=device, dtype=X.dtype),
        'cluster_neglogpdf':   torch.tensor(0.0, device=device, dtype=X.dtype)
    }
    
    
    # ====================================================================
    # GNN-Message Passing neural block
    time_GNN = time.time()
    Z = model['net'].encode(X, edge_index_hat)
    
    # ====================================================================
    # 2-point correlations neural block: from the latent space --> edge scores

    # Predict
    edge_logits = model['net'].decode(Z, edge_index_hat).view(-1)
    edge_prob   = edge_logits.sigmoid()
    time_GNN    = time.time() - time_GNN
    
    # ====================================================================
    
    if trainmode:
        gc.collect()
        
        loss['edge_BCE'], loss['edge_contrastive'], loss['track_neglogpdf'] = \
            hyperaux.edge_loss(model=model, Z=Z, x_ind=x_ind, edge_label=edge_label,
                      edge_index_hat=edge_index_hat,
                      objects=objects, edge_logits=edge_logits, edge_prob=edge_prob,
                      hit_weights=hit_weights, do_pdf=do_pdf, train_param=train_param, device=device)
    
    # ----------------------------------------------------------------
    # Before sparsification
    if DEBUG:
        tic  = time.time()
        data = torch_geometric.data.Data(edge_index=edge_index_hat, num_nodes=X.shape[0])
        G    = torch_geometric.utils.to_networkx(data)
        gg   = list(nx.weakly_connected_components(G))
        toc  = time.time() - tic
        cprint(__name__ + f'.hypertrack: Disconnected subgraphs before graph cut: {len(gg)} ({toc:0.3f} sec)', 'yellow')
    
    if device.type != 'cpu':
        iotools.showmem_cuda(device=device)
    else:
        iotools.showmem()
    
    # ================================================================
    # Clustering step
    
    time_CC      = 0.0
    time_cluster = 0.0
    
    cluster_labels_hat = None
    
    if do_clustering:
        
        # Sparsification
        sparse_ind  = (edge_prob > cluster_param['edge_threshold'])
        sparse_ind  = sparse_ind & (edge_index_hat[0,:] != edge_index_hat[1,:]) # non-self edges excluded by the second term
        
        cprint(__name__ + f'.hypertrack: Sparsification: {torch.sum(sparse_ind) / len(edge_prob):0.3f} (fraction of all edges) [cut = {cluster_param["edge_threshold"]}]', 'yellow')
        
        algo = cluster_param['algorithm']
            
        cprint(f'Clustering algorithm: <{algo}>', 'green')
        
        # Cast to CPU, transformer pivot search seems faster on CPU    
        edge_index_hat = edge_index_hat.detach().to('cpu')
        edge_prob      = edge_prob.detach().to('cpu')
        sparse_ind     = sparse_ind.detach().to('cpu')
        
        # ------------------------------------------------------------------------
        # Find disconnected subgraphs
        # https://math.stackexchange.com/questions/277045/easiest-way-to-determine-all-disconnected-sets-from-a-graph
        if  algo == 'cut' or algo == 'transformer':
            
            time_CC = time.time()
            
            # To Network-X object
            data  = torch_geometric.data.Data(edge_index=edge_index_hat[:,sparse_ind], num_nodes=X.shape[0])
            
            # Network-X CC search
            nxobj = torch_geometric.utils.to_networkx(data)
            gg_all  = list(nx.weakly_connected_components(nxobj))
            
            # CUGraph CC search on GPU (reservation)
            #G = cugraph.Graph()
            
            # Get size of each graph, sort (for transformer speed) and skip tiny (noise) graphs 
            idx     = np.argsort(np.array([len(gg_all[i]) for i in range(len(gg_all))], dtype=int))
            gg      = [list(gg_all[i]) for i in idx if len(gg_all[i]) >= cluster_param['min_graph']]
            
            time_CC = time.time() - time_CC
            print(__name__ + f'.hypertrack: Disconnected subgraphs after the edge cut and WCC search: {len(gg)} ({time_CC:0.3f} sec)')
        
        elif algo == 'dbscan' or algo == 'hdbscan':
            
            # To sparse CSR-matrix
            M = torchtools.to_scipy_sparse_csr_matrix(
                    edge_index = edge_index_hat[:, sparse_ind],
                    edge_attr  = 1.0 - edge_prob[sparse_ind].to(torch.float64), # distance metric as = 1 - p
                    num_nodes  = X.shape[0])
            
            # A technical trick to add one node, needed by hdbscan due to (possible) zero connections
            M = tools.attach_fully_connected_node(A=M, d=1.0)
            M = sklearn.neighbors.sort_graph_by_row_values(M, warn_when_not_sorted=False)
        
        # ------------------------------------------------------------------------
        
        time_cluster = time.time()
        
        ## ** 1. Cut based **
        if   algo == 'cut':
            output = {'trials': 1, 'failures': torch.tensor(0.0)}
            
            cluster_labels_hat, K = torchtools.cluster_list_to_tensor(lst=gg, N=X.shape[0])      
        
        ## ** 2. DBSCAN based **
        elif algo == 'dbscan':
            output = {'trials': 1, 'failures': torch.tensor(0.0)}

            # Cluster
            out = sklearn.cluster.DBSCAN(metric='precomputed', n_jobs=-1, **cluster_param['dbscan']).fit(M)
            out = np.array(out.labels_)
            
            K   = len(np.unique(out[out != -1]))
            cluster_labels_hat = torch.tensor(out, dtype=torch.long)
        
        ## ** 3. HDBSCAN based **
        elif algo == 'hdbscan':
            output = {'trials': 1, 'failures': torch.tensor(0.0)}
            
            # Cluster
            out = hdbscan.HDBSCAN(metric='precomputed', **cluster_param['hdbscan']).fit(M)
            out = out.labels_[:-1] # the last one is the virtual (extended) node
            
            K   = len(np.unique(out[out != -1]))
            cluster_labels_hat = torch.tensor(out, dtype=torch.long)
        
        ## ** 4. Transformer based **
        elif algo == 'transformer':
            
            if cluster_param['transformer']['threshold_algo'] == 'soft':
                cprint(__name__ + f'.hypertrack: Learnable node mask threshold: {float(model["net"].thr.data):0.6f}', 'yellow')
            
            output = transformer_helper(model=model, gg=gg, Z=Z, edge_prob=edge_prob, sparse_ind=sparse_ind,
                       edge_index_hat=edge_index_hat, objects=objects, X=X, hit_weights=hit_weights, true_clusters=true_clusters,
                       x_ind=x_ind, cluster_param=cluster_param, device=device,
                       train_param=train_param, trainmode=trainmode, do_pdf=do_pdf)

            cluster_labels_hat, K = torchtools.cluster_list_to_tensor(lst=output['cluster_hat'], N=X.shape[0])

        else:
            raise Exception(__name__ + f'.hypertrack: Unknown clustering algorithm chosen')
    
        time_cluster = time.time() - time_cluster
    
    if do_clustering:
        
        if trainmode and cluster_param['algorithm'] == 'transformer':
            
            # Collect losses
            loss = hyperaux.loss_normalization(loss=loss, output=output, K=K, train_param=train_param)
        
        aux = {
            'K': K,
            'failures':     int(torch.sum(output['failures'])),
            'trials':       output['trials'],  
            'time_GNN':     time_GNN,
            'time_CC':      time_CC,
            'time_cluster': time_cluster
        }
    else:
         aux = {
            'K':            0,
            'failures':     0,
            'trials':       0,
            'time_GNN':     time_GNN,
            'time_CC':      time_CC,
            'time_cluster': time_cluster 
        }

    # Other predictions
    aux['edge_prob'] = edge_prob.detach().cpu().to(torch.float32).numpy()
    
    return cluster_labels_hat, loss, aux

def transformer_helper(model, gg: List[List[int]], Z: torch.Tensor, edge_prob: torch.Tensor, sparse_ind: torch.Tensor,
                       edge_index_hat: torch.Tensor, objects: torch.Tensor,
                       X: torch.Tensor, hit_weights: torch.Tensor, true_clusters: torch.Tensor, x_ind: np.ndarray,
                       cluster_param: dict, train_param: dict, trainmode: bool, do_pdf: bool, device: torch.device):
    """
    Transformer clustering helper function
    """
    
    shared_obj = {
        'model':  model,
        'X':      X,
        'X_norm':   torch.norm(X, dim=-1),           # L2-norm
        'X_norm_T': torch.norm(X[:, 0:2], dim=-1),   # L2-transverse norm
        'Z':      Z,
        'hit_weights':    hit_weights,
        'true_clusters':  true_clusters,
        'objects':        objects,
        'x_ind':          x_ind}
    
    num_workers = cluster_param['worker_split']
    chunk_ind   = tools.split_start_end(range(len(gg)), num_workers)
    
    print(chunk_ind)
    
    output = None
    adjmat = torch_geometric.utils.to_dense_adj(
        edge_index=edge_index_hat[:,sparse_ind], edge_attr=edge_prob[sparse_ind], max_num_nodes=X.shape[0])[0]
    
    aux_time = time.time()
    
    for i in range(len(chunk_ind)):

        active_node_mask = []
        adjmat_local     = []
        global_ind       = []
        
        tic = time.time()
            
        for graph_index in range(chunk_ind[i][0], chunk_ind[i][1]):
            
            # Sort is necessary because we index linearly the (local) adjacency matrix
            global_ind__       = torch.sort(torch.tensor(list(gg[graph_index]),
                                                         dtype=torch.long, requires_grad=False, device=adjmat.device))[0]
            
            active_node_mask__ = torch.ones(len(global_ind__), dtype=torch.bool, requires_grad=False, device=adjmat.device)
            adjmat_local__     = adjmat[torch.meshgrid(global_ind__, global_ind__, indexing='ij')]
            
            # =========================================
            # Create subgraph objects
            
            global_ind.append(global_ind__)
            active_node_mask.append(active_node_mask__)
            adjmat_local.append(adjmat_local__)
            # =========================================

        aux_time = aux_time + (time.time() - tic)
            
        # Cluster this (set of) subgraph(s)
        if active_node_mask != []:
            
            output__ = clusterize(
                shared_obj, active_node_mask, adjmat_local, global_ind, train_param,
                cluster_param['transformer'], do_pdf, trainmode, device)
            
            if output is None:
                output = {}
                output['cluster_hat'] = output__['cluster_hat'].copy()
                
                for key in output__.keys():
                    if key != 'cluster_hat':
                        output[key] = output__[key].clone()
            else:
                for key in output.keys():
                    output[key] += output__[key]

    output['time_subgraphs'] = aux_time
    
    for key in output.keys():
        if 'time' in key:
            output[key] = float(output[key])
    
    return output

@torch.no_grad()
@torch.jit.script
def compute_topind(X_norm: torch.Tensor, X_norm_T: torch.Tensor,
                   nonzero:torch.Tensor, seed_ktop_max: int, seed_strategy: str):
    """
    Step 1. Compute seeding array of hits for pivotal search
    
    Args:
        X_norm:              3D-norm for each point
        X_norm_T:            2D-transverse norm for each point
        active_node_mask:    active nodes indices
        nonzero:             nonzero (active) indices
        seed_ktop_max:       maximum number of indices to pick in k-top
        seed_strategy:       seeding strategy
    
    Returns:
        seeding points
    """
    device = nonzero.device
    k      = min(seed_ktop_max, len(nonzero))
    ind    = torch.tensor(0, dtype=torch.int)
    
    if   seed_strategy == 'random':    # Random
        ind = torch.randperm(len(nonzero))[:k]

    elif seed_strategy == 'max':       # Max norm
        XX = X_norm[nonzero].squeeze()
        _, ind  = torch.topk(XX, k=k, largest=True)
    
    elif seed_strategy == 'min':       # Min norm
        XX = X_norm[nonzero].squeeze()
        _, ind  = torch.topk(XX, k=k, largest=False)
    
    elif seed_strategy == 'max_T':     # Max transverse norm
        XX = X_norm_T[nonzero].squeeze()
        _, ind  = torch.topk(XX, k=k, largest=True)
        
    elif seed_strategy == 'min_T':     # Min transverse norm
        XX = X_norm_T[nonzero].squeeze()
        _, ind  = torch.topk(XX, k=k, largest=False)
    
    return nonzero[ind.to(device)].squeeze()

@torch.no_grad()
@torch.jit.script
def diffuse_pivots_MC(topind: torch.Tensor, active_node_mask: torch.Tensor,
                          adjmat: torch.Tensor, N_pivots: int, threshold: float, random_paths: int, EPS:float=1E-3):
    """
    Step 2. Choose pivotal hit points based on seeding array (Monte Carlo Walk)
    
    Args:
        topind:             seed index candidate list
        active_node_mask:   active nodes
        adjmat:             adjacency matrix
        N_pivots:           number of pivots to be found
        threshold:          quality threshold
        random_paths:       number of MC random paths
        EPS:                denominator regularization
    """
    
    device      = adjmat.device
    best_pivots = [-1]
    best_weight = torch.tensor(-1E9, device=device)

    # MC repeat different paths
    for _ in range(random_paths):
        
        # Choose one of the pivots randomly
        ind    = torch.randint(low=0, high=len(topind), size=(1,))
        pivots = [int(topind[ind])]
        weight = torch.tensor(0.0, device=device)

        # Compute the path
        for n in range(1, N_pivots):
            
            row = active_node_mask * adjmat[pivots[n-1],:]
            row[pivots] = 0 # previous-pivots excluded

            # --------------------------------------------------------
            # Random sample according to the edge probabilities
            ii = torch.where(row > threshold)[0]
            p  = row[ii] / torch.sum(row[ii])
            if len(p) == 0:
                break # No further connections
            
            #j = p.multinomial(1)[0] # Pick one according to probabilities ~ p (slow)
            j = torchtools.fastcat_rand(p=p.to('cpu')) # Faster implementation
            action = int(ii[j])
            # --------------------------------------------------------
            
            pivots.append(action)
            weight += torch.log(row[action] / (1 - row[action] + EPS)) # Use logits
        
        if weight > best_weight:
            best_weight = weight
            best_pivots = pivots.copy()

    # ** Find the weakest (negative) connection **
    """
    if len(best_pivots) != 1:
        row = active_node_mask * adjmat[best_pivots[0],:]
        row[best_pivots] = 1E9 # positive-pivots excluded
        negative_pivot   = [int(torch.argmin(row))]
    else:
        negative_pivot = [-1]
    """
    negative_pivot = [-1]
    
    return best_pivots, negative_pivot


@torch.no_grad()
@torch.jit.script
def diffuse_pivots(topind: torch.Tensor, active_node_mask: torch.Tensor,
                   adjmat: torch.Tensor, N_pivots: int, threshold: float, EPS:float=1E-3):
    """
    Step 2. Choose pivotal hit points based on seeding array (Greedy Walk)
    
    Args:
        topind:             seed index candidate list
        active_node_mask:   active nodes
        adjmat:             adjacency matrix
        N_pivots:           number of pivots to be found
        threshold:          quality threshold
        EPS:                denominator regularization
    """

    device      = adjmat.device
    best_pivots = [-1]
    best_weight = torch.tensor(-1E9, device=device)
    
    # Start with each topind as the first pivot
    for s in range(len(topind)):
        
        # Choose the pivot
        pivots = [int(topind[s])]
        weight = torch.tensor(0.0, device=device)
        
        for n in range(1, N_pivots):
            
            row = active_node_mask * adjmat[pivots[n-1],:]
            row[pivots] = 0  # previous-pivots excluded
            k = torch.argmax(row)
            
            if row[k] > threshold:
                pivots.append(int(k))
                weight += torch.log(row[k] / (1 - row[k] + EPS)) # Use log-odds
                
            else:
                break
        
        # Check solution quality
        if weight > best_weight:
            best_weight = weight
            best_pivots = pivots.copy()
    
    # ** Find the weakest (negative) connection **
    """
    if len(best_pivots) != 1:
        row = active_node_mask * adjmat[best_pivots[0],:]
        row[best_pivots] = 1E9 # positive-pivots excluded
        negative_pivot   = [int(torch.argmin(row))]
    else:
        negative_pivot = [-1]
    """
    negative_pivot = [-1]
    
    return best_pivots, negative_pivot

@torch.no_grad()
@torch.jit.script
def extend_micrograph(adjmat: torch.Tensor, active_node_mask: torch.Tensor,
                      pivots: List[int], mode: str='pivot-spanned'):
    """
    Micrograph extension by picking all nodes which are connected to the pivots 
    
    Args:
        adjmat:           full adjacency matrix
        active_node_mask: active node array
        pivots:           pivotal node array
        mode:             'pivot-spanned' or 'full'
    
    Returns:
        connectivity mask
    """
    if mode == 'full':
        return active_node_mask.detach().squeeze()
    
    connect_mask = (adjmat[pivots[0],:] > 0) & active_node_mask
    
    for i in range(1, len(pivots)):
        mask = (adjmat[pivots[i],:] > 0) & active_node_mask
        connect_mask = mask | connect_mask # take OR
    
    return connect_mask.squeeze()

@torch.no_grad()
@torch.jit.script
def update_failures(pivots: List[int], active_node_mask: torch.Tensor, failures: torch.Tensor, max_failures: int):
    """
    Mask out graph nodes

    Args:
        pivots:             pivotal index list
        active_node_mask:   node mask
        failures:           number of failures tensor
        max_failures:       parameter for the maximum number of failures
    """

    failures[pivots] += 1
    for i in range(len(pivots)):    
        if failures[pivots[i]] >= max_failures:
            active_node_mask[pivots[i]] = 0

def clusterize(shared_obj:dict, active_node_mask: List[torch.Tensor], adjmat_local: List[torch.Tensor],
    global_ind: List[torch.Tensor], train_param: dict, cluster_param: dict, do_pdf: bool, trainmode: bool, device: torch.device):
    """
    Multi-Pivotal seeding and Set Transformer based clustering
    
    Args:
        shared_obj:        main data dictionary
        active_node_mask:  node mask
        adjmat_local:      local adjacency matrix
        global_ind:        global index list
        train_param:       training parameter dictionary
        cluster_param:     clustering parameter dictionary
        do_pdf:            evaluate normalizing flow
        trainmode:         training or inference mode
        device:            torch device
    
    Returns:
        clustering results and aux information in a dictionary
    """

    model          = shared_obj['model']
    X              = shared_obj['X']
    X_norm         = shared_obj['X_norm']
    X_norm_T       = shared_obj['X_norm_T']
    objects        = shared_obj['objects']
    
    Z              = shared_obj['Z']
    hit_weights    = shared_obj['hit_weights']
    true_clusters  = shared_obj['true_clusters']
    x_ind          = shared_obj['x_ind']

    # -----------------------------------------
    # Output
    output = {'cluster_hat':         [],
              'trials':              torch.tensor(0, dtype=torch.int),
              'failures':            torch.zeros(X.shape[0], dtype=torch.long),
              'cluster_BCE':         torch.tensor(0.0, device=device, dtype=X.dtype),
              'cluster_contrastive': torch.tensor(0.0, device=device, dtype=X.dtype),
              'cluster_neglogpdf':   torch.tensor(0.0, device=device, dtype=X.dtype)}
    
    # Which active_node_masks are not yet exhausted
    exhausted  = torch.zeros(len(active_node_mask), dtype=bool)
    
    pivot_seed_time    = 0
    pivot_diffuse_time = 0
    trf_time           = 0
    
    while True:
        
        # ==========================================================
        # ** Break the infinite loop -- we have exhausted every single active_node_mask **
        
        if torch.sum(exhausted) == len(exhausted):
            break
        
        # ==========================================================
        # Find pivots per graph
        
        all_pivots = len(active_node_mask)*[None]
        
        for b in range(len(active_node_mask)):
            if exhausted[b] == True:
                continue
            
            # ----------------
            tic = time.time()
        
            # Return non-zero node indices
            nonzero = torch.nonzero(active_node_mask[b])

            if len(nonzero) < cluster_param['N_pivots']: # Not enough data
                exhausted[b] = True
                continue
            
            # Pick index candidate array
            topind = compute_topind(X_norm=X_norm, X_norm_T=X_norm_T, nonzero=nonzero,
                seed_ktop_max=cluster_param['seed_ktop_max'],
                seed_strategy=cluster_param['seed_strategy'])
            
            pivot_seed_time += (time.time() - tic)

            # ----------------
            tic = time.time()
            
            # Diffuse pivots
            if cluster_param['random_paths'] > 1:
                
                pos_piv, neg_piv = diffuse_pivots_MC(topind=topind, active_node_mask=active_node_mask[b], adjmat=adjmat_local[b],
                    N_pivots=cluster_param['N_pivots'],
                    threshold=cluster_param['diffuse_threshold'],
                    random_paths=cluster_param['random_paths'])
            else:
                pos_piv, neg_piv = diffuse_pivots(topind=topind, active_node_mask=active_node_mask[b], adjmat=adjmat_local[b],
                    N_pivots=cluster_param['N_pivots'],
                    threshold=cluster_param['diffuse_threshold'])            
            
            if len(pos_piv) != cluster_param['N_pivots']:
                update_failures(pivots=pos_piv, active_node_mask=active_node_mask[b],
                        failures=output['failures'], max_failures=cluster_param['max_failures'])
            else:
                all_pivots[b] = pos_piv

            pivot_diffuse_time += (time.time() - tic)
        
        # ==========================================================
        # Create inclusive micrograph spanned by pivots (inclusive OR)
        
        tic = time.time()
        
        N              = np.zeros(len(active_node_mask), dtype=np.int64)
        micrograph_ind = len(active_node_mask) * [None]
        num_batch      = 0
        
        for b in range(len(active_node_mask)):
            if all_pivots[b] is not None:
                
                # Get micrograph mask
                connect_mask = extend_micrograph(adjmat=adjmat_local[b],
                                                    active_node_mask=active_node_mask[b],
                                                    pivots=all_pivots[b],
                                                    mode=cluster_param['micrograph_mode'])

                # List down active & connected indices in the global index space (!)
                micrograph_ind[b] = torch.nonzero(connect_mask).squeeze()
                N[b] = len(micrograph_ind[b])
                
                num_batch += 1

        if num_batch == 0: continue # Nothing found

        # ==========================================================
        # Create batched data for the transformer (all independent batches in parallel)
        # with gradients passing through from Z
        
        X_list       = []
        X_pivot_list = []
        
        for b in range(len(active_node_mask)):
            if all_pivots[b] is not None:
                
                X_list.append(torch.cat((Z[global_ind[b][micrograph_ind[b]],:],
                                         X[global_ind[b][micrograph_ind[b]],:]), -1))
                
                X_pivot_list.append(torch.cat((Z[global_ind[b][all_pivots[b]],:],
                                               X[global_ind[b][all_pivots[b]],:]), -1))
        
        X_fused, X_mask       = torchtools.batch_and_pad(sequences=X_list)
        X_pivot, X_pivot_mask = torchtools.batch_and_pad(sequences=X_pivot_list)
        
        # ==========================================================
        # Apply Transformer
        
        cluster_logits_all = model['net'].decode_cc_ind(X=X_fused, X_pivot=X_pivot, X_mask=X_mask, X_pivot_mask=X_pivot_mask)
        cluster_prob_all   = torch.sigmoid(cluster_logits_all)
        
        trf_time += (time.time() - tic)
        
        output['trials'] += 1
        
        # ==========================================================
        # Clustering per batch
        k = 0

        for b in range(len(active_node_mask)):
            if all_pivots[b] is not None:
                
                # ** Take batch result **
                cluster_logits  = cluster_logits_all[k][0:N[b]]
                cluster_prob    = cluster_prob_all[k][0:N[b]].view(-1)
                
                # -------------------------
                # Hard thresholding
                if cluster_param['threshold_algo'] != 'soft':
                    
                    # A. Adaptive Fisher (otsu) threshold
                    if cluster_param['threshold_algo'] == 'fisher':
                        
                        # Filter out (at maximum) top k candidates
                        param_k         = min(len(cluster_prob), cluster_param['ktop_max'])
                        toprob, topind  = torch.topk(cluster_prob, k=param_k, largest=True, sorted=False)
                        toprob          = toprob.detach().cpu().to(torch.float32).numpy()
                        topind          = topind.detach().cpu().numpy()    
                        
                        threshold  = losses.fisher_threshold(toprob, cluster_param['fisher_threshold'])
                        chosen_ind = micrograph_ind[b][topind[toprob > threshold]]
                        
                    # B. Fixed threshold (hyperparameter)
                    elif cluster_param['threshold_algo'] == 'fixed':
                        
                        threshold  = cluster_param['fixed_threshold']
                        chosen_ind = micrograph_ind[b][cluster_prob.to(micrograph_ind[b].device) > threshold]
                    else:
                        raise Exception(__name__ + '.hypertrack: Unknown threshold_algo chosen')
                    
                    chosen_ind_global = global_ind[b][chosen_ind]
                    
                # -------------------------
                # Learnable "soft" threshold via sigmoid relaxation
                # (gradients of cluster_prob are propagated through node set intersection loss)
                else:
                    
                    cluster_prob      = cluster_prob * torch.sigmoid((cluster_prob - model['net'].thr) / cluster_param['tau'])
                    chosen_ind        = micrograph_ind[b][cluster_prob.cpu() > float(model['net'].thr.data)]
                    chosen_ind_global = global_ind[b][chosen_ind]
                
                # =====================================================
                # Check feasibility
                if len(chosen_ind_global) < cluster_param['min_cluster_size']:
                    update_failures(pivots=all_pivots[b], active_node_mask=active_node_mask[b],
                                failures=output['failures'], max_failures=cluster_param['max_failures'])
                    continue
                
                # -----------------------------------------------------
                ### Now update the mask
                active_node_mask[b][chosen_ind] = False
                if torch.count_nonzero(active_node_mask[b]) < cluster_param['N_pivots']: exhausted[b] = True
                
                # Clustering output result
                output['cluster_hat'].append(chosen_ind_global.tolist())
                # -----------------------------------------------------

                # -----------------------------------------------------
                # TRAINING ONLY
                if trainmode:
                    
                    this_cluster_BCE, this_cluster_contrastive, this_cluster_neglogpdf = \
                        hyperaux.cluster_loss(model=model, Z=Z, global_ind=global_ind[b], chosen_ind=chosen_ind, pivots=all_pivots[b], \
                            micrograph_ind=micrograph_ind[b], cluster_logits=cluster_logits, cluster_prob=cluster_prob, \
                            x_ind=x_ind, true_clusters=true_clusters, hit_weights=hit_weights, train_param=train_param, \
                            objects=objects, do_pdf=do_pdf, device=device)

                    output['cluster_BCE']         = output['cluster_BCE']         + this_cluster_BCE
                    output['cluster_contrastive'] = output['cluster_contrastive'] + this_cluster_contrastive
                    output['cluster_neglogpdf']   = output['cluster_neglogpdf']   + this_cluster_neglogpdf
                # ==================================================
                
                k += 1 # !
    
    output['time_seed']    = torch.tensor(pivot_seed_time)
    output['time_diffuse'] = torch.tensor(pivot_diffuse_time)
    output['time_trf']     = torch.tensor(trf_time)
    
    print(__name__ + f".clusterize: Input graphs = {len(active_node_mask):4d} | Transformer trials = {output['trials']:4d} | Clusters = {len(output['cluster_hat']):4d} | pivot time = ({pivot_seed_time:0.3f},{pivot_diffuse_time:0.3f}) s | transformer time = {trf_time:0.3f} s")
    
    return output
