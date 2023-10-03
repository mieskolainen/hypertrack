# Evaluation metrics and measures
#
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np
import numba
from sklearn import metrics
from scipy import interpolate

from hypertrack.tools import select_valid

@numba.njit(parallel=True, fastmath=True)
def majority_score(x_ind_hat: np.ndarray, x_ind: np.ndarray, hit_weights: np.ndarray, hits_min: int=4):
    """
    Majority score function as defined in TrackML Kaggle challenge
    
    Args:
        x_ind_hat:    Estimated hit indices of tracks array
        x_ind:        Ground truth hit indices of tracks array
        hit_weights:  Weights of hits
        hits_min:     Minimum number of hits per track threshold
    
    Returns:
        score:        Double majority score value
        efficiency:   Efficiency = # (estimated AND matched) / # (ground truth with >= hits_min)
        purity:       Purity     = # (estimated AND matched) / # (estimated)
        passed:       Ground truth track indices which passed the double majority match_index
        match_index:  Pointing match indices per estimate --> to ground truth tracks
    
    See also: https://en.wikipedia.org/wiki/Jaccard_index
    """
    
    # Compare wrt ground truth
    passed   = np.zeros(x_ind.shape[0], dtype=np.bool_)
    scores   = np.zeros(x_ind_hat.shape[0], dtype=np.float32)
    match_index = (-1)*np.ones(x_ind_hat.shape[0], dtype=np.int64)
    
    for i in numba.prange(x_ind_hat.shape[0]):
        
        hit_set_hat = set(select_valid(x_ind_hat[i,:]))
        
        if len(hit_set_hat) < hits_min: continue # Too little hits
        
        # Loop over all ground truth
        its = (-1)*np.ones(x_ind.shape[0])
        
        for j in range(x_ind.shape[0]):
            
            if passed[j]: continue # Already used

            hit_set = set(select_valid(x_ind[j,:]))
            
            if len(hit_set) < hits_min: continue # Too little hits

            # Compute intersection set
            IS = hit_set.intersection(hit_set_hat)

            # Double majority criteria
            hit_purity     = len(IS) / len(hit_set_hat)
            hit_efficiency = len(IS) / len(hit_set)
            
            if hit_purity > 0.5 and hit_efficiency > 0.5: # > makes matching unique (not >=)
                ind    = np.array(list(IS))
                its[j] = np.sum(hit_weights[ind]) # sum of hit weights
        
        if np.sum(its > 0) > 0: # Pick the best matching estimate
            match_index[i] = np.argmax(its)
            scores[i]   = its[match_index[i]]

            # Tag the ground truth already used
            passed[match_index[i]] = 1
    
    ## Compute max score denominator
    max_score = 0.0
    N_true = 0
    for i in range(x_ind.shape[0]):
        
        hit_set = select_valid(x_ind[i,:])
        if len(hit_set) < hits_min: continue # Too little hits
        
        N_true += 1
        max_score += np.sum(hit_weights[hit_set])
    
    ## Compute efficiency and purity
    N_est_and_matched = np.sum(match_index != -1)
    efficiency = N_est_and_matched / N_true
    purity     = N_est_and_matched / len(match_index)
    
    return np.sum(scores) / max_score, efficiency, purity, passed, match_index


@numba.njit(fastmath=True)
def compute_adj_metrics(A: np.ndarray, A_hat: np.ndarray):
    """
    Compute metrics between adjacency matrices

    Args:
        A     : True adjacency (boolean)
        A_hat : Estimated adjacency (boolean)
    
    Returns:
        metrics as a tuple
    """
    tp = 0
    tn = 0
    fn = 0
    fp = 0

    pos = 0
    neg = 0
    
    N  = A.shape[0]

    for i in range(N):
        for j in range(N):
            if A[i,j] == True:
                pos += 1
            else:
                neg += 1

            if   A[i,j] == True  and A_hat[i,j] == True:
                tp += 1
            elif A[i,j] == False and A_hat[i,j] == False:
                tn += 1
            elif A[i,j] == True  and A_hat[i,j] == False:
                fn += 1
            elif A[i,j] == False and A_hat[i,j] == True:
                fp += 1

    acc = (tp + tn) / (pos + neg)
    pur = tp / (tp + fp)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    # Need to return simple tuple because of Numba
    return N, pos, neg, tp, tn, fp, fn, acc, pur, tpr, fpr

def adjgraph_metrics(A: np.ndarray, A_hat: np.ndarray, print_on: bool=True):
    """
    Compute adjacency graph comparison metrics
    
    Args:
        A:        True adjacency matrix (N x N)
        A_hat:    Estimated adjacency matrix (N x N)
        print_on: Print to to screen
    
    Returns:
        metrics dictionary
    """
    N, pos, neg, tp, tn, fp, fn, acc, pur, tpr, fpr = compute_adj_metrics(A=A, A_hat=A_hat)
    esum_A      = np.count_nonzero(A)
    esum_A_hat  = np.count_nonzero(A_hat)

    if print_on:
        print(__name__ + f'.print_graph_metrics:')
        print('')
        print(f'Ground Truth Adjacency (A)')
        print(f' Nodes N                   = {N}')
        print(f' Positive edges POS        = {pos}')
        print(f' Negative edges NEG        = {neg}')
        print(f' POS / NEG                 = {pos/neg:0.2E}')
        print('')
        print(f'Estimate (A_hat)')
        print(f' True Positive  TP         = {tp}')
        print(f' True Negative  TN         = {tn}')
        print(f' False Positive FP         = {fp}')
        print(f' False Negative FN         = {fn}')
        print(f' Accuracy                  = {acc:0.4f}    ACC = (TP + TN) / (POS + NEG)')
        print(f' Purity                    = {pur:0.4f}    PUR = TP / (TP + FP)')
        print(f' True  Positive Efficiency = {tpr:0.4f}    TPR = TP / POS = TP / (TP + FN)')
        print(f' False Positive Efficiency = {fpr:0.4f}    FPR = FP / NEG = FP / (FP + TN)')
        print('')
        print('Edge count')
        print(f' |A|           = {esum_A}')
        print(f' |A_hat|       = {esum_A_hat}')
        print(f' |A_hat| / |A| = {esum_A_hat:0.1E} / {esum_A:0.1E} = {esum_A_hat/esum_A:0.2f}')
        print(f' |A_hat| / N^2 = {esum_A_hat:0.1E} / {N**2:0.1E} = {esum_A_hat/N**2:0.2E}')
        print(f' |A| / N^2     = {esum_A:0.1E} / {N**2:0.1E} = {esum_A/N**2:0.2E}')
        print('')

    return {'N': N, 'pos': pos, 'neg': neg, 'tp': tp, 'tn': tn,
            'fp': fp, 'fn': fn, 'acc': acc, 'pur': pur, 'tpr': tpr, 'fpr': fpr,
            'esum_A': A, 'esum_A_hat': esum_A_hat}


@numba.njit(fastmath=True)
def true_false_positive(threshold_vector: np.ndarray, y_true: np.ndarray, sample_weight: np.ndarray=None):
    """
    ROC curve helper function
    
    Args:
        threshold_vector: thresholded values array (boolean)
        y_true:           true labels array (boolean)
        sample_weight:    weight per array element
    """
    
    if sample_weight is None:
        sample_weight = 1.0
    
    TP = np.sum(((threshold_vector == 1) & (y_true == 1)) * sample_weight)
    TN = np.sum(((threshold_vector == 0) & (y_true == 0)) * sample_weight)
    FP = np.sum(((threshold_vector == 1) & (y_true == 0)) * sample_weight)
    FN = np.sum(((threshold_vector == 0) & (y_true == 1)) * sample_weight)

    fpr = FP / (FP + TN)    
    tpr = TP / (TP + FN)

    return fpr, tpr


@numba.njit(fastmath=True)
def histedges_equal_N(x: np.ndarray, nbin: int):
    """
    Generate (histogram) bin edges such that each bin contains equal number of entries
    
    Args:
        x:     data points
        nbin:  number of bins
    
    Returns:
        histogram edges array
    """
    npt = len(x)
    return np.interp(x=np.linspace(0, npt, nbin + 1), xp=np.arange(npt), fp=np.sort(x))


@numba.njit(parallel=True, fastmath=True)
def roc_curve(y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray=None, points: int=256,
              sampling:str ='log10_reverse', th_type:str ='>=', epsilon: float=1E-7, drop_intermediate: bool=True):
    """
    Compute ROC curve
    
    Args:
        y_true:         True class labels (0,1) array
        y_score:        Predictor scores array
        sample_weight:  Sample weight array
        points:         Number of points on curve
        sampling:       'log10', 'linear', 'log10_reverse', or 'equalized'
        th_type:        '>=' or '>' used in the comparison
        epsilon:        Minimum value (in upper tail)
    
    Returns:
        fpr:  False positive rate array
        tpr:  True positive rate array
        thr:  Threshold array
    """
    
    # Shift to positive values
    shift   = np.abs(np.min(y_score)) + epsilon
    y_score = y_score + shift

    th_range = np.array([np.min(y_score), np.max(y_score)])
    
    tpr = np.zeros(points)
    fpr = np.zeros(points)
    
    # Sample from max to min
    thr = np.ones(points)
    
    if   sampling == 'equalized':
        thr = np.flip(histedges_equal_N(x=y_score, nbin=points))
    
    elif sampling == 'linear':
        thr = np.linspace(th_range[1], th_range[0], points)
    
    elif sampling == 'log10':
        thr = np.logspace(np.log10(th_range[1]), np.log10(th_range[0]), points)
    
    elif sampling == 'log10_reverse':
        thr = np.ones(points)*th_range[1] - np.logspace(np.log10(th_range[0]), np.log10(th_range[1]), points)
    
    # Threshold select
    if   th_type == '>':            
        for i in numba.prange(points):
            fpr[i], tpr[i] = true_false_positive(y_score > thr[i], y_true, sample_weight)
    elif th_type == '>=':
         for i in numba.prange(points):
            fpr[i], tpr[i] = true_false_positive(y_score >= thr[i], y_true, sample_weight)
    
    # Compensate shift
    thr = thr - shift
    
    # Compress via discrete curvature check
    if drop_intermediate and len(fpr) > 2:
        one = np.ones(1, dtype=np.bool_)
        optimal_idxs = np.where(np.concatenate((one, np.logical_or(np.diff(fpr, 2), np.diff(tpr, 2)), one)))[0]
        fpr = fpr[optimal_idxs]
        tpr = tpr[optimal_idxs]
        thr = thr[optimal_idxs]
    
    return fpr, tpr, thr


class Metric:
    """
    Classifier performance evaluation metrics.
    """
    def __init__(self, y_true, y_pred, weights=None, num_classes=2, hist=True, valrange='prob',
        N_mva_bins=30, verbose=True, num_bootstrap=0, roc_points=256):
        """
        Args:
            y_true     : true classifications
            y_pred     : predicted probabilities per class (N x 1), (N x 2) or (N x K) dimensional array
            weights    : event weights
            num_classes: number of classses
            
            hist       : histogram soft decision values
            valrange   : histogram range selection type
            N_mva_bins : number of bins
    
        Returns:
            metrics, see the source code for details
        """
        self.acc = -1
        self.auc = -1
        self.fpr = -1
        self.tpr = -1
        self.thresholds = -1
        
        self.tpr_bootstrap = None
        self.fpr_bootstrap = None
        self.auc_bootstrap = None
        self.acc_bootstrap = None

        self.mva_bins = []
        self.mva_hist = []

        # Make sure they are binary (scikit ROC functions cannot handle continuous values)
        y_true = np.round(y_true)

        self.num_classes = num_classes

        # Transform N x 2 to N x 1 (pick class[1] probabilities as the signal)
        if (num_classes == 2) and (np.squeeze(y_pred).ndim == 2):
            y_pred = y_pred[:,-1]

        # Make sure the weights array is 1-dimensional, not sparse array of (events N) x (num class K)
        if (weights is not None) and (np.squeeze(weights).ndim > 1):
            weights = np.sum(weights, axis=1)
        
        # Check numerical validity
        """
        if (np.squeeze(y_pred).ndim > 1):
            ok = np.isfinite(np.sum(y_pred,axis=1))
        else:
            ok = np.isfinite(y_pred)
        
        y_true = y_true[ok]
        y_pred = y_pred[ok]
        if weights is not None:
            weights = weights[ok]
        """
        
        # Invalid input
        if len(np.unique(y_true)) <= 1:
            if verbose:
                print(__name__ + f'.Metric: only one class present cannot evaluate metrics (return -1)')

            return # Return None
            
        if hist is True:
            
            # Bin the soft prediction values
            if   valrange == 'prob':
                valrange = [0.0, 1.0]
            elif valrange == 'auto':
                valrange = [np.percentile(y_pred, 1), np.percentile(y_pred, 99)]
            else:
                raise Exception('Metric: Unknown valrange parameter')

            self.mva_bins = np.linspace(valrange[0], valrange[1], N_mva_bins)
            self.mva_hist = []
            
            for c in range(num_classes):
                ind    = (y_true == c)
                counts = []
                
                if np.sum(ind) != 0:
                    w = weights[ind] if weights is not None else None
                    x = y_pred[ind] if num_classes == 2 else y_pred[ind,c]
                    counts, edges = np.histogram(x, weights=w, bins=self.mva_bins)
                self.mva_hist.append(counts)
        else:
            self.mva_bins = None
            self.mva_hist = None

        # ------------------------------------
        ## Compute Metrics
        out = compute_metrics(num_classes=num_classes, y_true=y_true, y_pred=y_pred, weights=weights, roc_points=roc_points)

        self.acc        = out['acc']
        self.auc        = out['auc']
        self.fpr        = out['fpr']
        self.tpr        = out['tpr']
        self.thresholds = out['thresholds']

        if num_bootstrap > 0:

            self.tpr_bootstrap = (-1)*np.ones((num_bootstrap, len(self.tpr)))
            self.fpr_bootstrap = (-1)*np.ones((num_bootstrap, len(self.fpr)))
            self.auc_bootstrap = (-1)*np.ones(num_bootstrap)
            self.acc_bootstrap = (-1)*np.ones(num_bootstrap)

            for i in range(num_bootstrap):

                # ------------------
                trials = 0
                max_trials = 10000
                while True:
                    ind = np.random.choice(range(len(y_true)), size=len(y_true), replace=True)
                    if len(np.unique(y_true[ind])) > 1 or trials > max_trials: # Protection with very low per class stats
                        break
                    else:
                        trials += 1
                if trials > max_trials:
                    print(__name__ + f'.Metric: bootstrap fail with num_classes < 2 (check the input per class statistics)')
                    continue
                # ------------------

                ww  = weights[ind] if weights is not None else None
                out = compute_metrics(num_classes=num_classes, y_true=y_true[ind], y_pred=y_pred[ind], weights=ww, roc_points=roc_points)
                
                if out['auc'] > 0:
                    
                    self.auc_bootstrap[i] = out['auc']
                    self.acc_bootstrap[i] = out['acc']

                    # Interpolate ROC-curve (re-sample to match the non-bootstrapped x-axis)
                    func = interpolate.interp1d(out['fpr'], out['tpr'], 'linear')
                    self.tpr_bootstrap[i,:] = func(self.fpr)

                    func = interpolate.interp1d(out['tpr'], out['fpr'], 'linear')
                    self.fpr_bootstrap[i,:] = func(self.tpr)


def compute_metrics(num_classes: int, y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, roc_points: int):
    """
    Helper function for Metric class
    """
    acc = -1
    auc = -1
    fpr = -1
    tpr = -1
    thresholds = -1
    
    # Fix NaN
    num_nan = np.sum(~np.isfinite(y_pred))
    if num_nan > 0:
        print(__name__ + f'.compute_metrics: Found {num_nan} NaN/Inf (set to zero)')
        y_pred[~np.isfinite(y_pred)] = 0 # Set to zero
    
    try:
        if  num_classes == 2:
            
            fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred, sample_weight=weights, points=roc_points)
            auc = np.trapz(y=tpr, x=fpr)
            acc = metrics.accuracy_score(y_true=y_true, y_pred=np.round(y_pred), sample_weight=weights)
        else:
            fpr, tpr, thresholds = None, None, None
            auc = metrics.roc_auc_score(y_true=y_true,  y_score=y_pred, sample_weight=None, \
                        average="weighted", multi_class='ovo', labels=np.arange(num_classes))
            acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred.argmax(axis=1), sample_weight=weights)
    except Exception as e:
        print(__name__ + f'.compute_metrics: Unable to compute ROC-metrics: {e}')

        import traceback, sys
        print("-"*60)
        traceback.print_exc(file=sys.stdout)
        print("-"*60)
        
        for i in range(num_classes):
            print(f'num_class[{i}] = {np.sum(y_true == i)}')

    return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': auc, 'acc': acc}
