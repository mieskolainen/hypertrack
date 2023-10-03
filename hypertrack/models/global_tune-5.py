# HyperTrack model and training loss parameters
# 
# match with the corresponding 'models_<ID>.py' under 'hypertrack/models/'
#
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np

# -------------------------------------------------------------------------
# Input normalization
# (e.g. can accelerate training, and mitigate float scale problems, but not necessarily needed)
normalize_input = False

"""
- coord[0] (min,max,mean,std): -1025.3399658203125 | 1025.3399658203125 | 1.0586246252059937 | 266.20428466796875
- coord[1] (min,max,mean,std): -1025.3399658203125 | 1025.3399658203125 | -0.022702794522047043 | 267.56085205078125
- coord[2] (min,max,mean,std): -2955.5 | 2955.5 | 1.6228374242782593 | 1064.4954833984375
"""

def feature_scaler(X):
    mu    = [1.06, -0.023, 1.62]
    sigma = [266.2, 267.6, 1064.5]
    
    for i in range(len(mu)):
        X[:,i] = (X[:,i] - mu[i]) / sigma[i]

# -------------------------------------------------------------------------

# ** Training only parameters **
train_param = {
    
    # Total loss weights per each individual loss
    'beta': {
        
        'net': {
            'edge_BCE'           : 0.2,    # 0.2
            'edge_contrastive'   : 1.0,    # 1.0
            'cluster_BCE'        : 0.2,    # 0.2
            'cluster_contrastive': 0.2,    # 1.0
            'cluster_neglogpdf':   0.0,    # [EXPERIMENTAL] (keep it zero)
        },
        
        'pdf': {
            'track_neglogpdf':     1.0,    # [EXPERIMENTAL]
        }
    },
    
    # Edge loss
    'edge_BCE': {
        'type':            'Focal',    # 'Focal', 'BCE', 'BCE+Hinge'
        'gamma':               1.0,    # For 'Focal' (entropy exponent)
        'delta':              0.05,    # For 'BCE+Hinge' (proportion)
        'remove_self_edges':  False,   # Remove self-edges
        'edge_balance':        True    # true/false edge balance unity re-weight
    },
    
    # Contrastive loss per particle
    'edge_contrastive': {
        'weights':  True,            # TrackML hit weights ok with this
        'type':   'softmax',
        'tau':      0.3,             # temperature (see: https://arxiv.org/abs/2012.09740, https://openreview.net/pdf?id=vnOHGQY4FP1)
        'sub_sample': 300,           # memory constraint (maximum number of target objects to compute the loss per event)
        
        'min_prob':  1e-3,           # minimum edge prob. score to be included in the loss [EXPERIMENTAL]
    },                               # (higher values push towards purity, but can weaken efficiency for e.g. high multiplicity clusters)
    
    # Cluster hit binary cross entropy loss
    'cluster_BCE': {
        'weights': False,            # TrackML hit weights (0 for noise) not exactly compatible
        'type':          'Focal',    # 'BCE', 'BCE+Hinge', 'Focal'
        'gamma':             1.0,    # For 'Focal' (entropy exponent)
        'delta':            0.05,    # For 'BCE+Hinge' (proportion)
    },
    
    # Cluster set hit loss
    'cluster_contrastive': {
        'weights':  False,        # TrackML hit weights (0 for noise) not exactly compatible
        'type':   'intersect',    # 'intersect', 'dice', 'jaccard'
        'smooth':    1.0          # regularization for 'dice' and 'jaccard'
    },
    
    # Cluster meta-supervision target
    'meta_target':  'pivotmajor'   # 'major' (vote from all nodes ground truth) or 'pivotmajor' (vote from pivots ground truth)
}

# -------------------------------------------------------------------------

# These algorithm parameters can be changed after training, but
# note that the transformer network may adapt (learn) its weights according
# to the values set here during the training
cluster_param = {

    # These are set from the command line interface
    'algorithm':            None,
    'edge_threshold':       None,

    
    ## Cut clustering & Transformer clustering input
    'min_graph':               4,     # Minimum subgraph size after the threshold and WCC search, the rest are treated as noise
    
    ## DBSCAN clustering
    'dbscan': {
        'eps':               0.2,     
        'min_samples':         3,
    },
    
    ## HDBSCAN clustering
    # https://hdbscan.readthedocs.io/en/latest/api.html
    'hdbscan': {
        'algorithm':         'generic',
        'cluster_selection_epsilon': 0.0,
        'cluster_selection_method': 'eom',  # 'eom' or 'leaf'
        'alpha':             1.0,
        'min_samples':         2,     # Keep it 2
        'min_cluster_size':    4,    
        'max_dist':           1.0     # Keep it 1.0
    },
    
    ## Transformer clustering
    'worker_split':              4,   # GPU Memory <-> GPU latency tradeoff (no accuracy impact)
    
    'transformer': {
        'seed_strategy':  'random',   # 'random', 'max' (max norm), 'max_T (transverse max), 'min' (min norm), 'min_T' (transverse min)
        'seed_ktop_max':         2,   # Number of pivot walk (seed) candidates (higher -> better accuracy but slower)
        
        'N_pivots':              3,   # Number of pivotal hits to search per cluster (>> 1)
        'random_paths':          1,   # (Put >> 1 for MC sampled random walk, and 1 for greedy max-prob walk)
        
        'max_failures':          2,   # Maximum number of failures per pivot list nodes (put 1+ for greedy, >> 1 for MC walk)
        'diffuse_threshold':   0.4,   # Diffusion connectivity ~ Pivot quality threshold
        
        # Micrograph extension type: 'pivot-spanned' (ok with 'hyper' adjacency), 'full' (for other than 'hyper' needed, more inclusive but possibly unstable)
        'micrograph_mode':   'pivot-spanned',
        
        'threshold_algo':    'fixed', # 'soft' (learnable), 'fisher' (batch-by-batch 1D-Fisher rule adaptive) or 'fixed'
        
        'tau':                 0.001, # 'soft':: Sigmoid 'temperature' (tau -> 0 ~ heaviside step)
        
        'ktop_max':           30,     # 'fisher':: Maximum cluster size (how many are considered from Transformer output), ranked by mask score
        'fisher_threshold': np.linspace(0.4,0.6, 0), # 'fisher':: Threshold values tested
        
        'fixed_threshold':     0.5,   # 'fixed':: Note, if this is too high -> training may be unstable (first transformer iterations are bad)
        
        'min_cluster_size':      4,   # Require at least this many constituents per cluster
    }
}


# -------------------------------------------------------------------------

### Geometric adjacency estimator
geom_param  = {
    
    # Use pre-trained 'voxdyn' or 'neurodyn' (experimental)
    'algorithm':  'voxdyn',
    
    # Print adjacency metrics (this will slow down significantly)
    'verbose': False,
    
    #'device':  'cuda', # CUDA not working with Faiss from conda atm (CUDA 11.4)
    'device': 'cpu',

    # 'neurodyn' parameters (PLACEHOLDER; not implemented)
    'neural_param': {
        'layers':  [6, 128, 64, 1],
        'act':     'silu',
        'bn':        True,
        'dropout':    0.0,
        'last_act':  False
    },
    
    'neural_path': 'models/neurodyn'
}


# -------------------------------------------------------------------------

### GNN + Transformer model parameters
net_model_param = {
    
    # GNN predictor block
    'graph_block_param': {
        
        'GNN_model' : 'SuperEdgeConv',   # 'SuperEdgeConv', 'GaugeEdgeConv'
        'nstack':           5,           # Number of GNN message passing layers
        
        'coord_dim':        3,           # Input dimension
        'h_dim':           64,           # Intermediate latent embedding dimension
        'z_dim':           61,           # Final latent embedding dimension
        
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#aggregation-operators
        'SuperEdgeConv': {
            'm_dim':                   64,
            'aggr':             ['mean']*5, # 'mean' (seems best memory/accuracy wise), 'sum', 'max', 'softmax', 'multi-aggregation', 'set-transformer'
            'use_residual':          True,
        },
        
        'GaugeEdgeConv': {
            'm_dim':                   64,
            'aggr':                  ['mean']*5, # As many as 'nstack'
            'norm_coord':             False,
            'norm_coord_scale_init':   1e-2,
        },
        
        # Edge prediction (correlation MLP) type: 'symmetric-dot', 'symmetrized', 'asymmetric'
        # (clustering Transformer should prefer 'symmetric-dot')
        'edge_type':   'symmetric-dot',
        
        ## Convolution (message passing) MLPs
        
        'MLP_GNN_edge': {
            'act':         'silu',       # 'relu', 'tanh', 'silu', 'elu'
            'bn':            True,
            'dropout':        0.0,
            'last_act':      True,
        },
        
        #'MLP_GNN_coord': {               # Only for 'GaugeEdgeConv'
        #    'act':         'silu',
        #    'bn':            True,
        #    'dropout':        0.0,
        #    'last_act':      True,
        #},

        'MLP_GNN_latent': {
            'act':         'silu',
            'bn':            True,
            'dropout':        0.0,
            'last_act':      True,
        },
        
        ## Latent Fusion MLP
        'MLP_fusion': {
            'act':         'silu',
            'bn':            True,
            'dropout':        0.0,
            'last_act':      True,
        },

        ## 2-pt edge correlation MLP
        'MLP_correlate': {
            'act':         'silu',
            'bn':            True,
            'dropout':        0.0,
            'last_act':     False,
        },
    },
    
    # Transformer clusterization block
    'cluster_block_param': {
        'in_dim':         64,       # Same as GNN 'zdim' + 3 (for 3D coordinates)
        'h_dim':          64,       # Latent dim, needs to be divisible by num_heads
        'output_dim':      1,       # Always 1
        'nstack_dec':      4,       # Number of self-attention layers
        
        'MLP_enc': {                # First encoder MLP
            'act':         'silu',  
            'bn':           False,
            'dropout':        0.0,
            'last_act':     False,
        },
        
        'MAB_dec': {                # Transformer decoder MAB
            'num_heads':        4,
            'ln':            True,
            'dropout':        0.0,
            'MLP_param':{
                'act':         'silu',  
                'bn':           False,
                'dropout':        0.0,
                'last_act':      True,
            }
        },
        
        'SAB_dec': {                # Transformer decoder SAB
            'num_heads':        4,
            'ln':            True,
            'dropout':        0.0,
            'MLP_param':{
                'act':         'silu',  
                'bn':           False,
                'dropout':        0.0,
                'last_act':      True,
            }
        },
        
        'MLP_mask': {               # Mask decoder MLP
            'act':         'silu',  
            'bn':           False,
            'dropout':        0.0,
            'last_act':     False,
        }
    }
}

# -------------------------------------------------------------------------
# [EXPERIMENTAL] -- normalizing flow

# Conditional data array indices (see /hypertrack/trackml.py)
cond_ind = [0,1,2,3,4,5,6]

pdf_model_param = {
    'in_dim':          61,
    'num_cond_inputs':  len(cond_ind),
    'h_dim':          196,
    'nblocks':          4,
    'act':          'tanh'
}
