# HyperTrack training aux functions
#
# m.mieskolainen@imperial.ac.uk, 2023

import os
import torch
import torch_geometric

import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import numpy as np
import glob
import gc

from termcolor import cprint

from hypertrack import hyper, trackml, tools, torchtools, visualize, iotools, measures


def set_optimizer(model: dict, learning_rate: float, args: dict):
    """
    Set Optimizer parameters
    
    Args:
        model:          different models under training by different optimizers
        learning_rate:  learning rate
        args:           setup parameters
    
    Returns:
        torch optimizer
    """
    
    optim_groups     = torchtools.seperate_weight_param(model=model['net'], weight_decay_w=args.weight_decay_w,     weight_decay_b=args.weight_decay_b)
    optim_groups_pdf = torchtools.seperate_weight_param(model=model['pdf'], weight_decay_w=args.weight_decay_pdf_w, weight_decay_b=args.weight_decay_pdf_b)
    
    if   args.optimizer == 'Adam':
        optimizer = {
            'net': torch.optim.Adam(params=optim_groups, lr=learning_rate, eps=args.epsilon),
            'pdf': torch.optim.Adam(params=optim_groups_pdf, lr=args.learning_rate_pdf, eps=args.epsilon_pdf)
        }
    elif args.optimizer == 'AdamW': # The best
        
        optimizer = {
            'net': torch.optim.AdamW(params=optim_groups, lr=learning_rate, eps=args.epsilon),
            'pdf': torch.optim.AdamW(params=optim_groups_pdf, lr=args.learning_rate_pdf, eps=args.epsilon_pdf)
        }
    elif args.optimizer == 'SGD':   # Minimum memory consumption (with momentum = 0)
        optimizer = {
            'net': torch.optim.SGD(params=optim_groups, lr=learning_rate, momentum=0),
            'pdf': torch.optim.SGD(params=optim_groups_pdf, lr=args.learning_rate_pdf, momentum=0)
        }
    else:
        raise Exception(f'Unknown optimizer {args.optimizer} chosen')
    
    return optimizer


def set_scheduler(optimizer: dict, args: dict):
    """
    Args:
        optimizer: optimizers for different models
        args:      setup parameters
    
    Returns:
        torch scheduler
    """
    
    if args.scheduler_type == 'cos':
        
        # "Cosine cycle scheme"
        lf = lambda x: (((1 + math.cos(x * math.pi / args.scheduler_period)) / 2) ** 1.0) * 0.9 + 0.1
        
        scheduler = {
            'net': torch.optim.lr_scheduler.LambdaLR(optimizer['net'], lr_lambda=lf),
            'pdf': torch.optim.lr_scheduler.LambdaLR(optimizer['pdf'], lr_lambda=lf)
        }

    elif args.scheduler_type == 'warm-cos':
        
        scheduler = {
            'net': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer['net'], T_0=args.scheduler_period, eta_min=args.learning_rate / 10),
            'pdf': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer['pdf'], T_0=args.scheduler_period, eta_min=args.learning_rate_pdf / 10)
        }
    
    elif args.scheduler_type == 'exp':
        
        gamma = 1 - 1e-5
        
        scheduler = {
            'net': torch.optim.lr_scheduler.ExponentialLR(optimizer['net'], gamma, last_epoch=-1),
            'pdf': torch.optim.lr_scheduler.ExponentialLR(optimizer['pdf'], gamma, last_epoch=-1)
        }
    
    elif args.scheduler_type == 'step':
        
        step_size = 1000
        gamma     = 0.9
        
        scheduler = {
            'net': torch.optim.lr_scheduler.StepLR(optimizer['net'], step_size, gamma),
            'pdf': torch.optim.lr_scheduler.StepLR(optimizer['pdf'], step_size, gamma)
        }
    
    elif args.scheduler_type == 'constant':
        
        factor      = 1.0
        total_iters = int(1E9)
        
        scheduler = {
            'net': torch.optim.lr_scheduler.ConstantLR(optimizer['net'], factor, total_iters),
            'pdf': torch.optim.lr_scheduler.ConstantLR(optimizer['pdf'], factor, total_iters)
        }
    
    else:
        raise Exception(__name__ + f'.set_scheduler: Unknown scheduler_type')
    
    return scheduler


def plot_metrics(epoch: int, metrics: dict, model: dict, args: dict, train_param: dict, alpha: float=0.5, filter_N: int=100):
    """
    Plot training metrics

    Args:
        epoch:       epoch (or iteration) vector
        metrics:     metrics dictionary
        model:       neural model (for gradient flow figures)
        args:        arguments object
        train_param: training parameters
        alpha:       transparency
        filter_N:    running average window length
    
    Returns:
        saves figures to the disk
    """
    CWD = os.getcwd()
    
    for folder in [f'{CWD}/figs/train/{args.save_tag}/metric', f'{CWD}/figs/train/{args.save_tag}/gradflow']:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    def func(x, filter):
        if filter:
            return tools.running_mean_uniform_filter1d(x, N=filter_N)
        else:
            return x
    
    # ------------------------------------------------------------------
    # Learning rate
    for scale in ['linear', 'log']:
        
        fig,ax = plt.subplots()
        plt.plot(metrics['learning_rate_net'], alpha=alpha, label=f'model [net]')
        plt.plot(metrics['learning_rate_pdf'], alpha=alpha, label=f'model [pdf]')

        plt.ylabel('learning rate')
        plt.xlabel('iteration')
        
        plt.legend(fontsize=8)
        
        ax.grid(True, which='major')
        ax.set_yscale(scale)
        ax.set_xscale(scale)
        ax.autoscale(enable=True, axis='both', tight=True)
        plt.savefig(f'{CWD}/figs/train/{args.save_tag}/metric/train_learning_rate--{scale}.pdf', bbox_inches='tight')
        plt.close()
    
    # ------------------------------------------------------------------
    # Net model
    for filter in [False, True]:
        for scale in ['linear', 'log']:
            
            fig,ax = plt.subplots()
            plt.plot(func(np.array(metrics['trn_edge_BCE']), filter),            alpha=alpha, label=f'$\\beta={train_param["beta"]["net"]["edge_BCE"]:0.2f}$ (edge BCE)')
            plt.plot(func(np.array(metrics['trn_edge_contrastive']), filter),    alpha=alpha, label=f'$\\beta={train_param["beta"]["net"]["edge_contrastive"]:0.2f}$ (edge contrastive)')
            plt.plot(func(np.array(metrics['trn_cluster_BCE']), filter),         alpha=alpha, label=f'$\\beta={train_param["beta"]["net"]["cluster_BCE"]:0.2f}$ (cluster BCE)')
            plt.plot(func(np.array(metrics['trn_cluster_contrastive']), filter), alpha=alpha, label=f'$\\beta={train_param["beta"]["net"]["cluster_contrastive"]:0.2f}$ (cluster contrastive)')
            
            plt.ylabel(f'train loss (without $\\beta$ applied)')
            plt.xlabel('iteration')
            plt.legend(fontsize=8)
            
            ax.grid(True, which='major')
            ax.set_yscale(scale)
            ax.set_xscale(scale)
            ax.autoscale(enable=True, axis='both', tight=True)
            plt.savefig(f'{CWD}/figs/train/{args.save_tag}/metric/train_losses_net--{scale}--filter-{filter}.pdf', bbox_inches='tight')
            plt.close()
    
    # ------------------------------------------------------------------
    # PDF model
    for filter in [False, True]:
        for scale in ['linear', 'log']:
            fig,ax = plt.subplots()
            plt.plot(func(np.array(metrics['trn_track_neglogpdf']), filter), alpha=alpha, label=f'$\\beta={train_param["beta"]["pdf"]["track_neglogpdf"]:0.2f}$ (track neglogpdf)')
            
            plt.ylabel(f'train loss (without $\\beta$ applied)')
            plt.xlabel('iteration')
            plt.legend(fontsize=8)
            
            ax.grid(True, which='major')
            ax.set_yscale(scale)
            ax.set_xscale(scale)
            ax.autoscale(enable=True, axis='both', tight=True)
            plt.savefig(f'{CWD}/figs/train/{args.save_tag}/metric/train_losses_pdf--{scale}--filter-{filter}.pdf', bbox_inches='tight')
            plt.close()

    # ------------------------------------------------------------------
    # Plot majority score
    
    for scale in ['linear', 'log']:
        
        fig,ax = plt.subplots()
        
        trn_score = metrics['trn_majority_score']
        plt.plot(trn_score,    label='train', alpha=alpha)
        plt.plot(func(trn_score, True), color='black', alpha=alpha, label='train [average]')
        val_score = metrics['val_majority_score']
        plt.plot(val_score,    label='validate', alpha=alpha)
        plt.plot(func(val_score, True), color='gray',  alpha=alpha, label='validate [average]')
        plt.ylabel('double majority score')
        plt.xlabel('iteration')
        plt.legend(fontsize=8)
        
        ax.set_yscale(scale)
        ax.set_xscale(scale)
        ax.grid(True, which='major')
        ax.autoscale(enable=True, axis='both', tight=True)
        plt.ylim([0.5, 1.0])
        plt.savefig(f'{CWD}/figs/train/{args.save_tag}/metric/majority_score--{scale}.pdf', bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Plot AUC
    
    for ymin in [0.8, 0.98]:
        for scale in ['linear', 'log']:
            
            fig,ax = plt.subplots()
            
            trn_auc = metrics['trn_edge_auc']
            plt.plot(trn_auc, label='train (edge)', alpha=alpha)
            plt.plot(func(trn_auc, True), color='black', alpha=alpha, label='train (edge) [average]')
            val_auc = metrics['val_edge_auc']
            plt.plot(val_auc, label='validate (edge)', alpha=alpha)
            plt.plot(func(val_auc, True), color='gray',  alpha=alpha, label='validate (edge) [average]')
            
            plt.ylabel('AUC')
            plt.xlabel('iteration')
            plt.legend(fontsize=8)
            
            ax.set_yscale(scale)
            ax.set_xscale(scale)
            ax.grid(True, which='major')
            ax.autoscale(enable=True, axis='both', tight=True)
            plt.ylim([ymin, 1.0])
            plt.savefig(f'{CWD}/figs/train/{args.save_tag}/metric/edge_AUC--{scale}_ymin_{100*ymin:0.0f}.pdf', bbox_inches='tight')
            plt.close()


def plot_gradient_flow(epoch:int, model:dict, args:dict):
    """
    Plot gradient flow
    
    Args:
        epoch:  epoch (or iteration) number
        model:  models in a dictionary
        args:   main parameters
    
    Returns:
        saves figures to the disk
    """
    CWD = os.getcwd()
    
    for m in model.keys():
        if model[m].training_on:
            all_figs = torchtools.plot_grad_flow(model[m].named_parameters(), param_types=[''])
            for i in range(len(all_figs)):
                plt.sca(all_figs[i]['ax'])
                plt.savefig(f"{CWD}/figs/train/{args.save_tag}/gradflow/{m}_gradflow_{all_figs[i]['name']}_tag_{args.save_tag}_epoch_{epoch}.pdf")
                plt.close(all_figs[i]['fig']);
            gc.collect() # Make sure no memory leak


def save_models(epoch: int, model: dict, optimizer: dict, scheduler: dict, metrics: dict, args: dict):
    """
    Save model objects to the disk

    Args:
        epoch:      epoch (or iteration)
        model:      torch models
        optimizer:  torch optimizers
        scheduler:  torch schedulers
        metrics:    metrics dictionary
        args:       setup parameters

    Returns:
        saves objects to the disk
    """
    CWD = os.getcwd()
    
    for folder in [f'{CWD}/models/tag_{args.save_tag}']:
        if not os.path.exists(folder):
            os.makedirs(folder)

    for key in model.keys():
        save_path = f'{CWD}/models/tag_{args.save_tag}/model_{key}_epoch_{epoch}.pt'
        print(__name__ + f'.save_models: Saving model to: {save_path}')
        checkpoint = { 
            'epoch':      epoch,
            'model':      model[key].state_dict(),
            'metrics':    metrics
        }
        torch.save(checkpoint, save_path)

    for key in optimizer.keys():
        save_path = f'{CWD}/models/tag_{args.save_tag}/optimizer_{key}_epoch_{epoch}.pt'
        print(__name__ + f'.save_models: Saving optimizer to: {save_path}')
        checkpoint = { 
            'epoch':      epoch,
            'optimizer':  optimizer[key].state_dict()
        }
        torch.save(checkpoint, save_path)

    for key in scheduler.keys():
        save_path = f'{CWD}/models/tag_{args.save_tag}/scheduler_{key}_epoch_{epoch}.pt'
        print(__name__ + f'.save_models: Saving scheduler to: {save_path}')
        checkpoint = { 
            'epoch':      epoch,
            'scheduler':  scheduler[key].state_dict()
        }
        torch.save(checkpoint, save_path)


def load_models(epoch: int, model: dict, args: dict, transfer_tag: str=None):
    """
    Load models from the disk
    
    Args:
        epoch:  epoch number (or iteration)
        model:  torch model(s)
        args:   setup parameters
    
    Returns:
        prev_metrics: the checkpoint metrics
        found_epoch   found epoch number
        
        Input 'model' is updated via reference
    """
    prev_metrics = None
    found_epoch  = None
    
    print(__name__ + f'.load_models')
    
    if transfer_tag is not None:
        load_tag = transfer_tag
        cprint(__name__ + f".load_optimizers: Transfer learning from the tag '{transfer_tag}'", 'magenta')
    else:
        load_tag = args.save_tag
    
    for key in model.keys():
        try:
            filename, found_epoch = iotools.grab_torch_file(key=key, name='model', save_tag=load_tag, epoch=epoch)
            
            checkpoint = torch.load(filename, map_location=args.device)
            # strict=False ignores missing keys (useful if changing the model structure)
            model[key].load_state_dict(checkpoint['model'], strict=False)
        
        except Exception as e:
            print(e)

        model[key].to(args.device)
        model[key].train() # Important!
        
        prev_metrics = checkpoint['metrics'] # Same for all model keys
        
        cprint(__name__ + f'.load_models: Loaded model [{key}]: {filename}', 'green')
    
    return prev_metrics, found_epoch


def load_optimizers(epoch: int, optimizer: dict, scheduler: dict, args: dict, transfer_tag: str = None):
    """
    Load optimizers from the disk
    
    Args:
        epoch:      epoch number (or iteration)
        optimizer:  torch optimizer(s)
        args:       setup parameters
    
    Returns:
        Input 'optimizer' is updated via reference
    """
    print(__name__ + f'.load_optimizers')
    
    if transfer_tag is not None:
        load_tag = transfer_tag
        cprint(__name__ + f".load_optimizers: Transfer learning from the tag '{transfer_tag}'", 'magenta')
    else:
        load_tag = args.save_tag
    
    for key in optimizer.keys():
        try:            
            path, found_epoch = iotools.grab_torch_file(key=key, name='optimizer', save_tag=load_tag, epoch=epoch)
            checkpoint = torch.load(path, map_location=args.device)
            optimizer[key].load_state_dict(checkpoint['optimizer'])
            
            # https://github.com/pytorch/pytorch/issues/80809
            for i in range(len(optimizer[key].param_groups)):
                optimizer[key].param_groups[i]['capturable'] = True
            
            cprint(__name__ + f'.load_models: Loaded optimizer [{key}]: {path}', 'green')
        except Exception as e:
            print(e)

    for key in scheduler.keys():
        try:
            path, found_epoch = iotools.grab_torch_file(key=key, name='scheduler', save_tag=load_tag, epoch=epoch)
            checkpoint = torch.load(path, map_location=args.device)
            scheduler[key].load_state_dict(checkpoint['scheduler'])

            cprint(__name__ + f'.load_models: Loaded scheduler [{key}]: {path}', 'green')
        except Exception as e:
            print(e)


def execute(trainmode: bool, event: int, rfactor: float,
            model: dict, do_clustering: bool, do_pdf: bool, args: dict, global_param: dict, geometric_obj: dict):
    """
    Execute HyperTrack training (or validation)
    
    Args:
        trainmode:      train or evaluation mode
        event:          event index
        rfactor:        pile-up reduction factor
        model:          model(s)
        do_clustering:  clustering on/off
        do_pdf:         pdf network on/off
        args:           setup parameters
        global_param:   global model parameters
        voronoi_obj:    geometric estimator object
        
    Returns:    
        total_loss:     total combined hybrid loss
        loss:           individual loss terms
        metric:         event metrics
    """
    CWD = os.getcwd()
    
    if   args.fp_dtype == 'float32':
        dtype = torch.float32
    elif args.fp_dtype == 'bfloat16':
        dtype = torch.bfloat16
    elif args.fp_dtype == 'float16':
        dtype = torch.float16
    
    if trainmode:

        cprint(__name__ + f'.execute: [Train | dtype = {dtype}] | do_clustering = {do_clustering}', 'red')
        for m in model.keys():
            if model[m].training_on:
                model[m].train()
    else:
        cprint(__name__ + f'.execute: [Validate | dtype = {dtype}] do_clustering = {do_clustering}', 'red')
        for m in model.keys():
            model[m].eval()
    
    # -------------------------------------------------------
    ## Generate data
    
    d = trackml.generate_graph_data(event=event, rfactor=rfactor, noise_ratio=args.noise_ratio, \
        node2node=args.node2node, geom_param=global_param.geom_param, geometric_obj=geometric_obj)

    X              = d['X']
    hit_weights    = d['hit_weights']
    
    #A              = d['A']             # Full Ground Truth Adjacency, not needed here at all
    #edge_index     = d['edge_index']    # --|--
    
    x_ind          = d['x_ind']          # Ground truth cluster constituents
    edge_label     = d['edge_label']
    
    edge_index_hat = d['edge_index_hat'] # Estimated graph adjacency
    #A_hat          = d['A_hat']         # Matrix version, Not needed
    
    # Cluster object ground truth physical properties (e.g. particle momentum ...)
    objects        = d['objects'][:, global_param.cond_ind]
    
    # -------------------------------------------------------
    ## Convert to torch
    
    
    objects        = torch.tensor(objects,        dtype=dtype,      device=args.device)
    X              = torch.tensor(X,              dtype=dtype,      device=args.device)
    edge_index_hat = torch.tensor(edge_index_hat, dtype=torch.long, device=args.device)
    edge_label     = torch.tensor(edge_label,     dtype=dtype,      device=args.device)
    hit_weights    = torch.tensor(hit_weights,    dtype=dtype,      device=args.device)

    # Feature normalization
    if global_param.normalize_input:
        print(__name__ + f'.normalizing input X')
        global_param.feature_scaler(X)
    
    # ** SORT (needed by certain message passing) **
    cprint(__name__ + '.execute: Sorting edge indices', 'red')
    edge_index_hat, edge_label = torch_geometric.utils.sort_edge_index(edge_index=edge_index_hat, edge_attr=edge_label, sort_by_row=False)
    
    # Create cluster labels
    if do_clustering:
        true_clusters = torch.tensor(tools.create_cluster_labels(N=X.shape[0], x_ind=x_ind), dtype=torch.long, device=args.device)
    else:
        true_clusters = None
    
    cprint(__name__ + f'.execute: [Event: {event:05d}, num_nodes = {X.shape[0]:3E}, num_edges = {edge_index_hat.shape[1]:3E}]', 'green')
    
    # -------------------------------------------------------
    # Main execute
    
    def wrap_call():
        
        # Runs the forward pass with autocasting
        #with torch.autocast(device_type=args.device, dtype=dtype):
        
        cluster_labels_hat, loss, aux = \
            hyper.hypertrack(
                model          = model,
                X              = X,
                edge_index_hat = edge_index_hat,
                x_ind          = x_ind,
                hit_weights    = hit_weights,
                edge_label     = edge_label,
                true_clusters  = true_clusters,
                objects        = objects,
                do_pdf         = do_pdf,
                do_clustering  = do_clustering,
                trainmode      = trainmode,
                cluster_param  = global_param.cluster_param,
                train_param    = global_param.train_param,
                device         = args.device)
        
        return cluster_labels_hat, loss, aux
        
    if trainmode:
        cluster_labels_hat, loss, aux = wrap_call()
    else:
        with torch.no_grad():   
            cluster_labels_hat, loss, aux = wrap_call()
    
    # -------------------------------------------------------
    # Metrics
    
    # Compare predicted (out) with the ground truth labels (edge_label)
    tic = time.time()
    
    try:
        y_true = edge_label.cpu().to(torch.float32).numpy()
        y_pred = aux['edge_prob']

        # Pick a random subset of edges to improve time spent
        subset = np.random.choice(len(y_true), min(len(y_true), int(2E5)), replace=False)
        met    = measures.Metric(y_true=y_true[subset], y_pred=y_pred[subset])
        
        if np.random.rand() < 0.01: # Save time, plot only X% of the time
            
            # Hit weights
            fig, ax = plt.subplots()
            hw = hit_weights.cpu().to(torch.float32).numpy()
            plt.hist(hw / np.max(hw), bins=50, density=True)
            plt.xlabel('hit weight'); plt.ylabel('density')
            plt.savefig(f'{CWD}/figs/train/{args.save_tag}/metric/hit_weights.pdf', bbox_inches='tight')
            plt.close()

            # MVA scores
            self_ind    = (edge_index_hat[0,:] == edge_index_hat[1,:]).cpu().numpy()
            nonself_ind = (edge_index_hat[0,:] != edge_index_hat[1,:]).cpu().numpy()
            
            visualize.density_MVA_wclass(y_pred=y_pred[self_ind], y=y_true[self_ind],
                                        label='$HyperTrack$ (GNN: self edges)', weights=None,
                                        num_classes=2, hist_edges=100, filename=f'{CWD}/figs/train/{args.save_tag}/metric/edge_score_self')

            visualize.density_MVA_wclass(y_pred=y_pred[nonself_ind], y=y_true[nonself_ind],
                                        label='$HyperTrack$ (GNN: non-self edges)', weights=None,
                                        num_classes=2, hist_edges=100, filename=f'{CWD}/figs/train/{args.save_tag}/metric/edge_score_nonself')
    
    except Exception as e:
        cprint(e, 'red')
        auc = 0

    time_roc = time.time() - tic

    # Compute majority score
    tic = time.time()

    DMS_score  = 0.0
    DMS_eff    = 0.0
    DMS_purity = 0.0
    
    if do_clustering:
        hmax       = 50 # array bound (maximum number of members per cluster)
        x_ind_hat  = tools.create_cluster_ind(cluster_labels_hat.cpu().numpy(), hmax=hmax)
        DMS_score, DMS_eff, DMS_purity, _, _ = measures.majority_score(x_ind=x_ind, x_ind_hat=x_ind_hat, hit_weights=hit_weights.cpu().to(torch.float32).numpy())
    
    time_majority_score = time.time() - tic

    metric = {'edge_auc': met.auc, 'edge_metrics': met, 'majority_score': DMS_score, 'efficiency': DMS_eff, 'purity': DMS_purity}
    
    cprint(f'Input Nodes = {X.shape[0]:6d}, Edges = {edge_index_hat.shape[1]:7d}, Objects = {x_ind.shape[0]:5d}', 'magenta')
    cprint(f'GNN: {aux["time_GNN"]:0.3f} s | CC: {aux["time_CC"]:0.3f} s | Clustering: {aux["time_cluster"]:0.3f} s | AUC = {met.auc:0.4f} ({time_roc:0.3f} s) | DMS = {DMS_score:0.3f} ({time_majority_score:0.3f} s) | Efficiency = {DMS_eff:0.3f} | Purity = {DMS_purity:0.3f} | Estimate nclust = {aux["K"]:5d} | Pivot failures = {aux["failures"]:4d}', 'magenta')
    print('')
    
    # -------------------------------------------------------
    # Collect losses for different independent optimizers
    
    total_loss = {}
    for model_key in global_param.train_param['beta'].keys():
        total_loss[model_key] = 0.0
    
    ### Over models
    for model_key in global_param.train_param['beta'].keys():
        print(f'[{model_key}]: | ', end='')
        
        # Weighted sum over loss terms
        for key in global_param.train_param['beta'][model_key].keys():
            
            total_loss[model_key] = total_loss[model_key] + global_param.train_param['beta'][model_key][key] * loss[key]
            
            print(f'{key} = {loss[key].item():0.3f} ({global_param.train_param["beta"][model_key][key]:0.2f}) | ', end='')
        print('')
    print('')
    
    return total_loss, loss, metric


@tools.timing
def backward(optimizer: dict, scheduler: dict, model: dict, total_loss: dict, scaler: dict,
             max_grad_norm: float=1.0, dtype=torch.float32):
    """
    Backward (backpropagation and scheduler update) evolution
    
    Args:
        optimizer:      torch optimizers
        scheduler:      torch schedulers
        model:          torch models
        total_loss:     losses per model
        scaler:         gradient scaler (for float16)
        max_grad_norm:  gradient norm clipping
        dtype:          floating point type
    """
    
    for m in model.keys():
        model[m].zero_grad(set_to_none=True) # set_to_none=True is more efficient
    
    # Check do we need to retain graph due to accessing the gradients by different optimizers
    retain_graph = False
    
    k = 0
    for m in total_loss.keys():
        if model[m].training_on and total_loss[m] is not None:
            k += 1
    if k > 1:
        retain_graph = True

    for j, m in enumerate(total_loss.keys()):

        if j == k - 1:
            retain_graph = False # Can release after the last

        if model[m].training_on and total_loss[m] is not None:
            
            # float32 and bfloat16
            if dtype != torch.float16:
                total_loss[m].backward(retain_graph=retain_graph)
                torch.nn.utils.clip_grad_norm_(model[m].parameters(), max_norm=max_grad_norm)
                optimizer[m].step()
            
            # float16
            else:
                scaler[m].scale(total_loss[m]).backward(retain_graph=retain_graph)
                
                scaler[m].unscale_(optimizer[m])
                torch.nn.utils.clip_grad_norm_(model[m].parameters(), max_norm=max_grad_norm)
                
                scaler[m].step(optimizer[m])
                scaler[m].update()
            
            if scheduler[m] is not None:
                scheduler[m].step()
                print(__name__ + f'.backward: [{m}]: Loss = {total_loss[m].item():0.3f} | Scheduler [{m}] learning rate = {scheduler[m].get_last_lr()[0]:0.3E}')
            else:
                print(__name__ + f'.backward: [{m}]: Loss = {total_loss[m].item():0.3f}')
        else:
            print(f'Optimizer [{m}] not active')

    