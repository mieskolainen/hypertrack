# Torch tools
#
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np
import torch
import scipy
import random
from prettytable import PrettyTable
from matplotlib import pyplot as plt

from typing import Optional, List
from torch import Tensor


def torch_to_scipy_scr(x, is_sparse=True):
    """
    Convert a dense or sparse torch tensor to a scipy csr matrix
    
    Args:
        x:          tensor to convert
        is_sparse:  sparse or normal tensor
    
    Returns:
        csr array
    """
    if is_sparse:
        values  = x._values()
        indices = x._indices()
        return scipy.sparse.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=x.shape)
    else:
        indices = x.nonzero().t()
        values  = x[indices[0], indices[1]]
        return scipy.sparse.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=x.shape)


def seperate_weight_param(model: torch.nn.Module, weight_decay_w: float, weight_decay_b: float):
    """
    Construct weight decays for different parameter types of the model
    
    Args:
        model:            torch model
        weight_decay_w:   weight decay value for layer weight terms
        weight_decay_b:   weight decay value for layer bias terms
    
    Returns:
        network parameters grouped 
    """
    
    table = PrettyTable(["Parameter", "Weight decay"])
    
    # Separate out all parameters to weight decay regulated (and not)
    # Typically one wants zero (0) weight decay for .bias and LayerNorm type terms
    decay    = set()
    no_decay = set()
    
    for pn, p in model.named_parameters():
        if pn.endswith('bias'):
            no_decay.add(pn)
            table.add_row([pn, weight_decay_b])
        elif pn.endswith('weight') and ('.ln' in pn or '.LayerNorm' in pn):
            no_decay.add(pn)
            table.add_row([pn, weight_decay_b])
        else:
            decay.add(pn)
            table.add_row([pn, weight_decay_w])
    
    print(table)
    
    # Validate parameter sets
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} appear in both decay (no_decay) sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} did not appear in decay (no_decay) set!"
        
    # Create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))],    "weight_decay": weight_decay_w},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": weight_decay_b},
    ]
    
    return optim_groups

def to_scipy_sparse_csr_matrix(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]=None, num_nodes: int=None) -> scipy.sparse.csr_matrix:
    """
    Converts a graph given by edge indices and edge attributes to a scipy
    sparse matrix in CSR format
    
    Args:
        edge_index:  adjacency list
        edge_attr:   edge features
        num_nodes:   number of nodes in the graph (do set, otherwise guess)
    
    Returns:
        sparse adjacency matrix (scipy csr)
    """
    row, col = edge_index.cpu()

    if edge_attr is None:
        edge_attr = torch.ones(row.size(0))
    else:
        edge_attr = edge_attr.view(-1).cpu()
        assert edge_attr.size(0) == row.size(0)

    #from torch_geometric.utils.num_nodes import maybe_num_nodes
    #N = maybe_num_nodes(edge_index, num_nodes)
    N = num_nodes
    out = scipy.sparse.csr_matrix((edge_attr.numpy(), (row.numpy(), col.numpy())), (N, N))

    return out

def weighted_degree(index: torch.Tensor, weight: torch.Tensor, num_nodes: int = None):
    """
    Weighted (graph) degree
    
    Args:
        index:      adjacency list row, i.e. edge_index[0,:]
        weight:     edge weights
        num_nodes:  number of nodes in the graph (do set, otherwise guess)
    
    Returns:
        weighted node degree
    """
    #from torch_geometric.utils.num_nodes import maybe_num_nodes
    #N = maybe_num_nodes(index, num_nodes)
    N = num_nodes
    out = torch.zeros((N, ), dtype=weight.dtype, device=index.device)
    
    return out.scatter_add_(0, index, weight)

def set_diff_1d(t1:torch.Tensor, t2:torch.Tensor, assume_unique=False):
    """
    Set difference of two 1D tensors.
    Returns the unique values in t1 that are not in t2.

    """
    if not assume_unique:
        t1 = torch.unique(t1)
        t2 = torch.unique(t2)
    return t1[(t1[:, None] != t2).all(dim=1)]

def torch_intersect(t1:torch.Tensor, t2:torch.Tensor, use_unique=False):
    """
    Set intersect A-B between two tensors
    
    Args:
        t1:  tensor A
        t2:  tensor B
        
    Returns:
        torch tensor for A-B
    """
    t1 = t1.unique()
    t2 = t2.unique()
    
    return torch.tensor(np.intersect1d(t1.cpu().numpy(), t2.cpu().numpy()))

def subgraph_mask(subgraph_nodes: List[int], edge_index_hat: torch.Tensor, sparse_ind=None):
    """
    Compute subgraph boolen mask array

    Args:
        subgraph_nodes:  list of subgraph nodes
        edge_index_hat:  edge index list (2 x E)
        sparse_ind:      already sparsified indices (dim E) (if none, all edges are treated)
    
    Returns:
        subgraph mask indices
    """
    nodes  = torch.tensor(subgraph_nodes, device=edge_index_hat.device)
    mask_a = torch.isin(edge_index_hat[0,:], nodes)
    mask_b = torch.isin(edge_index_hat[1,:], nodes)
    
    if sparse_ind is not None:
        return mask_a & mask_b & sparse_ind
    else:
        return mask_a & mask_b

@torch.jit.script
def cluster_list_to_tensor(lst: List[List[int]], N: int):
    """
    A list of clusters to node assignments tensor
    
    Args:
        lst: a list of lists of cluster nodes
        N:   the number of nodes in a graph
    
    Returns:
        arr: a cluster index assignment array (-1 set for non-clustered)
        K:   the number of non-empty clusters
    """
    
    arr = (-1)*torch.ones(N, dtype=torch.long)
    K = 0
    for i in range(len(lst)):
        if len(lst[i]) > 0: # Do not add empty []
            arr[list(lst[i])] = K
            K += 1
    
    return arr, K

@torch.jit.script
def batch_and_pad(sequences: List[torch.Tensor]):
    """
    Create a zero padded and batched tensor from a list of tensors
    
    This conserves gradient information
    """
    max_len    = max([s.shape[0] for s in sequences])
    out_dims   = (len(sequences), max_len, sequences[0].shape[-1])
    out_tensor = sequences[0].new_zeros(size=out_dims)
    mask       = torch.ones((len(sequences), max_len), dtype=torch.bool, device=sequences[0].device)
    
    for i, in_tensor in enumerate(sequences):
        N = in_tensor.shape[0]
        out_tensor[i, :N, :] = in_tensor
        mask[i, N:] = False
    
    return out_tensor, mask

def count_parameters_torch(model: torch.nn.Module, only_trainable: bool = False, print_table: bool=True):
    """
    Count the number of (trainable) pytorch model parameters
    
    Args:
        model:           torch model
        only_trainable:  if true, only which require grad are returned
        print_table:     print the parameter table
    
    Returns:
        number of parameters
    """
    
    table = PrettyTable(["Modules", "Parameters", "Trainable"])
    tot_trainable   = 0
    tot_untrainable = 0
    
    for name, parameter in model.named_parameters():
        
        params = parameter.numel()
        table.add_row([name, params, parameter.requires_grad])
        
        if parameter.requires_grad:
            tot_trainable   += params
        else:
            tot_untrainable += params
    
    if print_table:
        print(table)
        print(f"Total trainable params: {tot_trainable} | Total non-trainable params: {tot_untrainable} | Total: {tot_trainable+tot_untrainable}")
    
    if only_trainable:
        return tot_trainable
    else:
        return tot_trainable + tot_untrainable

def fastcat_rand(p: torch.Tensor):
    """
    Categorical (multinomial) sampling function
    
    Args:
        p: multinomial probability vector
    
    Return:
        random index
    """
    return (p.cumsum(-1) >= torch.rand(p.shape[:-1])[..., None]).byte().argmax(-1)


def plot_grad_flow(named_parameters, y_max=0.1, y_min=1e-12, alpha=0.2, param_types=['']):
    """
    Plots the gradients flowing through different layers in the net during training.
    Use for checking for possible gradient vanishing / explosion / debugging.
    
    Usage:
        Use in torch code after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    
    Args:
        named_parameters: as returned by model.named_parameters()
        y_max:            plot y-axis maximum ('auto', None, or float value)
        y_min:            y-axis minimum
        alpha:            bar plot transparency
        param_type:       a list of names to be found from the parameters, e.g. 'weight', 'bias', set [''] for all
    
    Returns:
        figures as a list
    """
    all_figs = []
    all_ax   = []

    for ptype in param_types:
        
        table = PrettyTable(["Parameter", "min |grad|", "avg |grad|", "max |grad|"])
        
        fig, ax = plt.subplots(figsize=(8,6))
        fig.subplots_adjust(bottom=0.4) 
        plt.sca(ax)
        
        avg_grads = []
        max_grads = []
        min_grads = []
        layers    = []
        
        for pname, p in named_parameters:
            if p.grad is None:
                #print(f'Parameter {n} has grad = None')
                continue
            
            if p.requires_grad and (ptype in pname or ptype == ''):
                layers.append(pname)
                avg_grads.append(p.grad.abs().mean().detach().cpu().to(torch.float32).numpy())
                max_grads.append(p.grad.abs().max().detach().cpu().to(torch.float32).numpy())
                min_grads.append(p.grad.abs().min().detach().cpu().to(torch.float32).numpy())
                
                table.add_row([f'{layers[-1]}', f'{min_grads[-1]:0.2E}', f'{avg_grads[-1]:0.2E}', f'{max_grads[-1]:0.2E}'])
        
        print(table)
        
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=alpha, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), avg_grads, alpha=alpha, lw=1, color="b")
        plt.bar(np.arange(len(max_grads)), min_grads, alpha=alpha, lw=1, color="r")
        
        plt.hlines(0, 0, len(avg_grads)+1, lw=0.5, color="k" )
        
        plt.xticks(range(0,len(avg_grads), 1), layers, rotation="vertical", fontsize=3)
        plt.xlim(left=0, right=len(avg_grads))
        
        if y_max == 'auto':
            plt.ylim(bottom=y_min, top=np.median(min_grads)) # zoom in on the lower gradient regions
        else:
            plt.ylim(bottom=y_min, top=y_max)
        
        plt.ylabel("$|\\nabla_w f(\\mathbf{x}, w)|$")
        plt.title(f"Gradient flow (.{ptype})")
        plt.yscale('log')
        
        plt.grid(True, alpha=0.5)
        plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                    plt.Line2D([0], [0], color="b", lw=4),
                    plt.Line2D([0], [0], color="r", lw=4)], ['max', 'mean', 'min'],
                    fontsize=7)
        
        all_figs.append({'name': ptype, 'fig': fig, 'ax': ax})

    return all_figs

def set_random_seed(seed):
    """
    Set random seed for torch, random and numpy
    
    Args:
        seed:  seed value
    """
    print(__name__ + f'.set_random_seed: Setting random seed <{seed}>')
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
