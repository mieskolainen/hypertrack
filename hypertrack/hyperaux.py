# Aux training functions for HyperTrack
#
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np
import torch
from typing import Tuple, List

from hypertrack import losses, tools

def edge_loss(model: dict, Z: torch.Tensor, x_ind: np.ndarray, edge_label: torch.Tensor,
                edge_index_hat: torch.Tensor, objects: torch.Tensor,      
                edge_logits: torch.Tensor, edge_prob: torch.Tensor,
                hit_weights: torch.Tensor,
                do_pdf: bool, train_param: dict, device: torch.device):
    """
    GNN output loss computation
    
    Args:
        model:             model dictionary
        Z:                 GNN latents
        x_ind:             cluster ground truth assignments
        edge_label:        edge ground truth labels
        edge_index_hat:    adjancency list
        objects:           clustering ground truth objects
        edge_logits:       GNN edge logits
        edge_prob:         GNN edge probabilities
        hit_weights:       hit weights
        do_pdf:            evaluate normalizing flow [experimental]
        train_param:       training parameters dictionary
        device:            torch device
    
    Returns:
        torch losses (tuple)
    """
    
    # --------------------------------------------------------------
    ## Edge loss
    
    # Remove self-edges
    valid = torch.ones(edge_label.shape, dtype=torch.bool)
    if train_param['edge_BCE']['remove_self_edges']:
        valid = (edge_index_hat[0,:] != edge_index_hat[1,:])
    
    # Edge class balance inverse weights
    edge_weight = torch.ones(edge_label.shape, dtype=edge_logits.dtype, device=device)
    if train_param['edge_BCE']['edge_balance']:
        ratio = torch.sum(edge_label[valid] == 0) / torch.sum(edge_label[valid] == 1)
        edge_weight[edge_label == 1] *= ratio
    
    if   train_param['edge_BCE']['type'] == 'BCE':
        
        edge_lossfunc = torch.nn.BCEWithLogitsLoss(weight=edge_weight[valid], reduction='mean')
        loss_edge = edge_lossfunc(edge_logits[valid], edge_label[valid])
    
    elif train_param['edge_BCE']['type'] == 'Focal':
        
        gamma = train_param['edge_BCE']['gamma']
        edge_lossfunc = losses.FocalWithLogitsLoss(weight=edge_weight[valid], gamma=gamma, reduction='mean')
        loss_edge = edge_lossfunc(edge_logits[valid], edge_label[valid])
    
    elif train_param['edge_BCE']['type'] == 'BCE+Hinge':
        edge_lossfunc = torch.nn.BCEWithLogitsLoss(weight=edge_weight[valid], reduction='mean')
        loss_edge = edge_lossfunc(edge_logits[valid], edge_label[valid])
        
        edge_lossfunc = losses.HingeLoss(weight=edge_weight[valid], reduction='mean')
        delta = train_param['edge_BCE']['delta']
        # note this needs {-1,1} labels
        loss_edge = (1-delta)*loss_edge + delta * edge_lossfunc(edge_logits[valid], 2*edge_label[valid] - 1)
    
    # --------------------------------------------------------------
    ## Contrastive loss
    
    if train_param['edge_contrastive']['weights']:
        weights = hit_weights
    else:
        weights = None
    
    # Cut-off regulator
    pick_ind = edge_prob > train_param['edge_contrastive']['min_prob']
    
    # We use .tanh() for logits to avoid numerical scale problems
    loss_edge_contrastive = losses.contrastive_edge_loss( \
        edge_score=edge_logits[pick_ind].tanh(),
        edge_index_hat_np=edge_index_hat[:, pick_ind].cpu().numpy(),
        x_ind=x_ind,
        hit_weights=weights,
        tau=train_param['edge_contrastive']['tau'],
        sub_sample=train_param['edge_contrastive']['sub_sample'])
    
    # --------------------------------------------------------------
    ## Normalizing flow PDF (EXPERIMENTAL)
    
    loss_track_neglogpdf = torch.tensor(0.0, device=device)
    
    if do_pdf:
        
        tot = torch.tensor(0.0, device=device)

        # Over each ground truth particle
        for i in range(x_ind.shape[0]):

            # Pick hit indices
            ind = tools.select_valid(x_ind[i,:])
            
             # Cluster object parameters, e.g. (3-vertex, 3-momentum, charge ...)
            conditional_data = objects[i,:].repeat(len(ind), 1)

            # Compute log-likelihood for each hit to originate given (conditional) the track parameters
            lnL = model['pdf'].track_pdf.log_probs(Z[ind, :], conditional_data).mean()
            
            tot = tot + lnL

        loss_track_neglogpdf = - tot / x_ind.shape[0]

    return loss_edge, loss_edge_contrastive, loss_track_neglogpdf


def cluster_loss(model: dict, Z: torch.Tensor, global_ind: torch.Tensor, chosen_ind: torch.Tensor, pivots: List[int],
                 micrograph_ind: torch.Tensor, cluster_logits: torch.Tensor, cluster_prob: torch.Tensor,
                 x_ind: np.ndarray, true_clusters: torch.Tensor, hit_weights: torch.Tensor,
                 objects: torch.Tensor, train_param: dict, do_pdf: bool, device: torch.device):
    """
    Transformer output loss computation

        Args:
            model:              model dictionary
            Z:                  GNN latents
            global_ind:         global index list
            chosen_ind:         chosen indices after threshold cut
            pivots:             pivot indices
            micrograph_ind:     micrograph indices
            cluster_logits:     transformer output logits
            cluster_prob:       transformer output probabilities
            x_ind:              cluster ground truth assignments
            true_clusters:      cluster labels
            hit_weights:        hit (nodes) weights
            objects:            cluster object data
            train_param:        training parameter dictionary
            do_pdf:             evaluate normalizing flow (experimental)
            device:             torch device
        
        Returns:
            losses (tuple)
    """
    
    ## Meta-supervision target label
    
    # WAY 1: Choose the one which is the major among chosen ones
    if   train_param['meta_target'] == 'major':
        class_number,_ = torch.mode(true_clusters[global_ind[chosen_ind]])
    
    # WAY 2: Choose the one which is the major among pivots
    elif train_param['meta_target'] == 'pivotmajor':
        class_number,_ = torch.mode(true_clusters[global_ind[pivots]])
    else:
        raise Exception(__name__ + f'.hypertrack: Unknown meta_target = {train_param["meta_target"]}')
    
    all_class_label = torch.zeros((len(true_clusters), 1), dtype=cluster_logits.dtype, device=device)
    all_class_label[true_clusters == class_number] = 1
    
    class_label = all_class_label[global_ind[micrograph_ind]]

    # --------------------------------------------------------------
    # Node weights
    if train_param['cluster_BCE']['weights']:
        weight = hit_weights[global_ind[micrograph_ind]][:,None]
        weight = weight / torch.sum(weight)
    else:
        weight = torch.ones_like(cluster_logits)    
        weight = weight / len(weight)
    
    ## Node mask loss
    if   train_param['cluster_BCE']['type'] == 'BCE':
        
        lossfunc = torch.nn.BCEWithLogitsLoss(weight=weight, reduction='sum')
        cluster_BCE = lossfunc(cluster_logits, class_label)
    
    elif train_param['cluster_BCE']['type'] == 'Focal':
        gamma = train_param['cluster_BCE']['gamma']    
        lossfunc = losses.FocalWithLogitsLoss(weight=weight, gamma=gamma, reduction='sum')
        cluster_BCE = lossfunc(cluster_logits, class_label)
    
    elif train_param['cluster_BCE']['type'] == 'BCE+Hinge':
        lossfunc = torch.nn.BCEWithLogitsLoss(weight=weight, reduction='sum')
        cluster_BCE = lossfunc(cluster_logits, class_label)
        
        lossfunc = losses.HingeLoss(weight=weight, reduction='sum')
        delta = train_param['cluster_BCE']['delta']
        # note this needs {-1,1} labels
        cluster_BCE = (1-delta)*cluster_BCE + delta*lossfunc(cluster_logits, 2*class_label - 1)
    
    # --------------------------------------------------------------
    ## Cluster set loss
    if   train_param['cluster_contrastive']['type'] == 'dice':
        lossfunc = losses.DiceLogitsLoss(weight=None, reduction='sum')
        cluster_contrastive = lossfunc(cluster_logits, class_label, smooth=train_param['cluster_contrastive']['smooth'])

    elif train_param['cluster_contrastive']['type'] == 'jaccard':
        lossfunc = losses.JaccardLogitsLoss(weight=None, reduction='sum')
        cluster_contrastive = lossfunc(cluster_logits, class_label, smooth=train_param['cluster_contrastive']['smooth'])

    elif train_param['cluster_contrastive']['type'] == 'intersect':
        
        if   train_param['cluster_contrastive']['weights']:
            weight = hit_weights / torch.sum(hit_weights)
        else:
            weight = torch.ones_like(hit_weights) / len(hit_weights)
        
        # True hit indices of the particle given by 'class_number'
        true_ind = tools.select_valid(x_ind[class_number,:])
        
        cluster_contrastive = losses.intersect_loss(prob=cluster_prob, weight=weight,
            true_ind=true_ind, chosen_ind=global_ind[chosen_ind], micrograph_ind=global_ind[micrograph_ind])

    else:
        raise Exception(__name__ + '.hypertrack: Unknown cluster_loss chosen')
    
    # --------------------------------------------------------------
    # Log-likelihood density model evaluation [EXPERIMENTAL TEST]
    # (currently this pull gradients only through GNN and not Transformer -- could take its latent output)
    if train_param['beta']['net']['cluster_neglogpdf'] > 0:
        
        # Sample random indices
        random_ind  = torch.randint(low=0, high=len(global_ind), size=(len(micrograph_ind),))
        
        data_random = Z[global_ind[random_ind], :]
        data        = Z[global_ind[micrograph_ind], :]
        
        conditional_data = objects[class_number,:].repeat(len(micrograph_ind), 1)
        
        model['pdf'].eval()
        
        # Random
        rand_lnL = model['pdf'].track_pdf.log_probs(data_random, conditional_data).sum()

        # Chosen
        true_lnL = model['pdf'].track_pdf.log_probs(data, conditional_data).sum()
        
        if model['pdf'].training_on:
            model['pdf'].train()
        
        # Use the negative log[likelihood ratio] as the loss
        cluster_neglogpdf = - (true_lnL - rand_lnL)
        
    else:
        cluster_neglogpdf = torch.tensor(0.0)
    
    return cluster_BCE, cluster_contrastive, cluster_neglogpdf


def loss_normalization(loss: dict, output: dict, K: int, train_param: dict):
    """
    The final loss normalization
    
    Args:
        loss:         loss dictionary
        output:       transformer output dictionary
        K:            number of clusters found
        train_param:  training parameters
    
    Returns:
        loss dictionary
    """
    
    # Cluster cardinality normalization, to avoid bias towards small number of clusters
    loss['cluster_BCE'] = output['cluster_BCE'] / K

    if   train_param['cluster_contrastive']['type'] == 'dice':
        loss['cluster_contrastive'] = output['cluster_contrastive'] / K

    elif train_param['cluster_contrastive']['type'] == 'jaccard': 
        loss['cluster_contrastive'] = output['cluster_contrastive'] / K
    
    elif train_param['cluster_contrastive']['type'] == 'intersect':     # Here we do not use 1 / K
        loss['cluster_contrastive'] = output['cluster_contrastive'] + 1 # + 1 is just a constant shift
    
    loss['cluster_neglogpdf'] = output['cluster_neglogpdf'] / K

    return loss
