# Loss functions
#
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np
import numba
import torch
from torch import nn

from hypertrack import tools


@torch.compile()
def intersect_loss(prob: torch.Tensor, true_ind: torch.Tensor, chosen_ind: torch.Tensor,
                   micrograph_ind: torch.Tensor, weight: torch.Tensor):
    """
    Intersect set loss
    
    Args:
        prob:           per node probability scores [0,1]
        true_ind:       ground truth indices
        chosen_ind:     chosen indices after the treshold cut
        micrograph_ind: indices of the micrograph
        weight:         weights per node
    
    Returns:
        torch loss
    """
    
    # Compute the intersection set between the ground truth and the hard threshold chosen estimate
    set_true = set(true_ind.tolist())
    set_hat  = set(chosen_ind.tolist())
    
    IS_pos   = list(set_true.intersection(set_hat)) # Were chosen right
    IS_neg   = list(set_hat - set_true)             # Were chosen wrong
    
    # Compute weights
    weight__         = torch.zeros(len(weight), dtype=weight.dtype, device=prob.device)
    weight__[IS_pos] =  weight[IS_pos]            # Give positive weights for the true indices
    weight__[IS_neg] = -weight[IS_neg]            # Give negative (penalty) weights for not true
    weight__         = weight__[micrograph_ind]   # Finally evaluate within indices
    
    # Loss
    return - torch.sum(weight__ * prob)

@numba.njit(parallel=True, fastmath=True)
def find_posneg_connections(i: int, N: int, x_ind: np.ndarray, edge_index_hat: np.ndarray, take_self_connections: bool=False):
    """
    Find all connections to ground truth particle
    
    Args:
        i:              current row index in x_ind
        N:              number of nodes (hits)
        x_ind:          truth particle hit assigments
        edge_index_hat: GNN input graph (hard) adjacency of the event
    
    Returns:
        e_pos:   positive connections i.e. (i,j) both found in true_ind
        e_neg:   negative (fake) connections i.e. i (but not j) or j (but not i) found in true_ind
        hit_ind: ground truth hit indices
    """
    
    true_ind_bool = np.zeros(N, dtype=np.bool_)
    
    hit_ind = tools.select_valid(x_ind[i,:])
    true_ind_bool[hit_ind] = True
    
    nedges = edge_index_hat.shape[1]    
    e_pos  = np.zeros(nedges, dtype=np.bool_)
    e_neg  = np.zeros(nedges, dtype=np.bool_)
    
    for n in numba.prange(nedges):
        
        i,j = edge_index_hat[0,n], edge_index_hat[1,n]
        a,b = true_ind_bool[i], true_ind_bool[j]
        
        # No connection
        if not a and not b:
            continue
        
        # Positive connections
        elif   a and b:
            if   i != j:
                e_pos[n] = True
            elif i == j and take_self_connections:
                e_pos[n] = True
        
        # Negative connections (a and not b) or (not a and b)
        else:
            e_neg[n] = True
    
    return e_pos, e_neg, hit_ind


@tools.timing
def contrastive_edge_loss(edge_score: torch.Tensor, edge_index_hat_np: np.ndarray,
                          x_ind: np.ndarray, hit_weights: torch.Tensor, tau:float = 0.1, sub_sample:int = 100000):
    """
    Supervised contrastive loss function based on ground truth objects and edges between hits
    
    Args:
        edge_score:        GNN scores per edge
        edge_index_hat_np: edge indices as a numpy array
        x_ind:             cluster ground truth indices
        hit_weights:       weights for each graph node
        tau:               temperature hyperparameter
        sub_sample:        maximum number of cluster objects to consider
    
    Returns:
        torch loss
    """
    
    loss    = 0.0
    N       = len(hit_weights)
    inv_tau = 1.0 / tau
    total_weight = 0.0
    
    rind   = np.random.permutation(min(sub_sample, x_ind.shape[0]))
    hitsum = 0.0
    k      = 0
    
    # Loop over all the objects
    for i in rind:
        
        # Construct positive and negative connections
        e_pos, e_neg, hit_ind = find_posneg_connections(i=i, N=N, x_ind=x_ind, edge_index_hat=edge_index_hat_np)

        # ** Compute loss via softmax **
        # One-by-one, each positive edge per object compared against all negative edges, then mean over
        if np.sum(e_pos) > 0:
            
            if hit_weights is not None:
                W = torch.sum(hit_weights[hit_ind])
            else:
                W = 1.0
            
            total_weight += W
            
            num   = torch.exp(edge_score[e_pos] * inv_tau)
            denom = num + torch.sum(torch.exp(edge_score[e_neg] * inv_tau))
            nll   = - torch.mean(torch.log(num / denom)) # i.e. 1 / num. of positive edges
            
            # Add to the total loss weighted by the importance of the object
            loss = loss + W * nll

            # For the overall loss scale normalization
            # (this can be a problem or e.g. vertex degree dependent issue)
            hitsum += len(hit_ind)
            k      += 1
    
    # Mean hit multiplicity
    hitsum /= k
    
    return loss / total_weight / hitsum


class FocalWithLogitsLoss(nn.Module):
    """
    Focal Loss with logits as input
    
    https://arxiv.org/abs/1708.02002
    """
    def __init__(self, weight=None, gamma=2, reduction="mean"):
        super(FocalWithLogitsLoss, self).__init__()
        self.weight    = weight
        self.bce       = torch.nn.functional.binary_cross_entropy_with_logits
        self.gamma     = gamma
        self.reduction = reduction
    
    def forward(self, predicted, target):
        pt           = torch.exp(-self.bce(predicted, target, reduction='none'))
        entropy_loss = self.bce(predicted, target, weight=self.weight, reduction='none')
        focal_loss   = ((1-pt)**self.gamma)*entropy_loss
        
        if   self.reduction == "none":
            return focal_loss
        elif self.reduction == "mean":
            if self.weight is None:
                return focal_loss.mean()
            else:
                return focal_loss.sum() / torch.sum(self.weight)
        else:
            return focal_loss.sum()

class SmoothHingeLoss(torch.nn.Module):
    """
    Smooth Hinge Loss (Neurocomputing 463 (2021) 379-387)
    """
    def __init__(self, weight=None, reduction='sum', sigma=0.1):
        """
        In the limit sigma -> 0, this recovers HingeLoss
        """
        super(SmoothHingeLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.sigma = sigma

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        # note this needs {-1,1} targets, i.e. multiply (0,1) targets by two and subtract 1
        
        arg  = 1.0 - torch.mul(output, target)
        loss = 0.5*arg + 0.5*torch.sqrt(arg**2 + self.sigma**2)
        
        if self.weight is not None:
            loss = self.weight * loss

        if   self.reduction == 'sum':
            return torch.sum(loss)
        if   self.reduction == 'mean':
            if self.weight is None:
                return torch.mean(loss)
            else:
                return torch.sum(loss) / torch.sum(self.weight)
        elif self.reduction == 'none':
            return loss


class HingeLoss(torch.nn.Module):
    """
    https://en.wikipedia.org/wiki/Hinge_loss
    """
    def __init__(self, weight=None, reduction='sum'):
        super(HingeLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        # note this needs {-1,1} targets, i.e. multiply (0,1) targets by two and subtract 1
        
        loss = 1.0 - torch.mul(output, target)
        loss[loss < 0] = 0
        
        if self.weight is not None:
            loss = self.weight * loss

        if   self.reduction == 'sum':
            return torch.sum(loss)
        if   self.reduction == 'mean':
            if self.weight is None:
                return torch.mean(loss)
            else:
                return torch.sum(loss) / torch.sum(self.weight)
        elif self.reduction == 'none':
            return loss

@numba.njit(parallel=True, fastmath=True)
def fisher_threshold(prob: np.ndarray, threshold_array: np.ndarray):
    """
    Find Fisher variance criterion (Otsu's method) optimal bimodal threshold
    
    Args:
        prob:  probabilities to threshold
        threshold_array: threshold to test values between [0,1]
    
    Returns:
        optimal threshold (float)
    """
    loss = np.zeros(len(prob), dtype=np.float32)
    for i in numba.prange(len(loss)):
        loss[i] = fisher_loss(prob, threshold_array[i])
    k = np.argmin(loss)
    
    if threshold_array[k] == np.inf: # Most likely unimodal input
        return 0.5
    else:
        return threshold_array[k]

@numba.njit(fastmath=True)
def fisher_loss(x: np.ndarray, th: float):
    """
    Fisher variance criterion loss for bimodal classifier threshold optimization
    
    Args:
        x:  array of data to be thresholded
        th: threshold value
    
    Returns:
        loss value (float)
    """
    thresholded_x = np.zeros(x.shape, dtype=np.float32)
    thresholded_x[x > th] = 1
    
    # Compute weights
    nb_pixels  = len(x)
    nb_pixels1 = np.count_nonzero(thresholded_x)
    weight1    = nb_pixels1 / nb_pixels
    weight0    = 1 - weight1

    # If one the classes is empty / all elements belong
    # to the other class, do not consider the threshold value
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # Find all elements belonging to each class
    val_pixels1 = x[thresholded_x == 1]
    val_pixels0 = x[thresholded_x == 0]

    # Compute variance of the classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    return weight0*var0 + weight1*var1

class JaccardLogitsLoss(nn.Module):
    """
    Intersect over Union set loss
    """
    def __init__(self, weight=None, reduction='sum'):
        super(JaccardLogitsLoss, self).__init__()
        self.weight = weight
        if   reduction == 'sum':
            self.sum = torch.sum
        elif reduction == 'mean':
            self.sum = torch.mean
        else:
            raise Exception('IoULogitsLoss: Unknown reduction chosen')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: float=1):

        inputs = torch.sigmoid(inputs)       
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Weights not implemented
        #w = 1.0 if self.weight is None else self.weight.view(-1) / torch.sum(self.weight.view(-1))

        # Intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions 
        intersection = self.sum(inputs * targets)
        total = self.sum(inputs + targets)
        union = total - intersection 
        
        IoU = (intersection + smooth) / (union + smooth)
        
        return 1 - IoU

class DiceLogitsLoss(nn.Module):
    """
    Dice set loss
    """
    def __init__(self, weight=None, reduction='sum'):
        super(DiceLogitsLoss, self).__init__()    
        
        self.weight = weight
        
        if   reduction == 'sum':
            self.sum = torch.sum
        elif reduction == 'mean':
            self.sum = torch.mean
        else:
            raise Exception('DiceLogitsLoss: Unknown reduction chosen')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: float=1):
        
        inputs = torch.sigmoid(inputs)       
        
        # Flatten label and prediction tensors
        inputs  = inputs.view(-1)
        targets = targets.view(-1)
        
        # Weights not implemented
        #w = 1.0 if self.weight is None else self.weight.view(-1) / torch.sum(self.weight.view(-1))

        intersection = self.sum(inputs * targets)                            
        dice = (2.0*intersection + smooth) / (self.sum(inputs) + self.sum(targets) + smooth)  
        
        return 1 - dice