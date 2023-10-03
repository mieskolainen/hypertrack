# HyperTrack neural model training code
#
# m.mieskolainen@imperial.ac.uk, 2023

import sys
sys.path.append("../hypertrack")

import os
import os.path as osp
import time
import numpy as np
import numba
import traceback
import random
import pickle
import gc
from termcolor import cprint

import csr
import torch
import importlib
from argparse import ArgumentParser

from hypertrack import auxtrain, iotools, torchtools, visualize

def main():
    CWD = os.getcwd()
    numba.config.THREADING_LAYER = 'tbb' # requires tbb to be installed

    for folder in [f'{CWD}/models', f'{CWD}/figs']:
        if not os.path.exists(folder):
            os.makedirs(folder)
    # --------------------------------------------------------------------
    
    
    parser = ArgumentParser()
    
    parser.add_argument("-o",        "--optimizer",          default='AdamW', type=str,   action='store', help="Optimizer {Adam, AdamW, SGD}")
    parser.add_argument("-lr",       "--learning_rate",      default=5e-4,    type=float, action='store', help="Learning rate (1st phase with GNN on)")
    parser.add_argument("-lr2",      "--learning_rate_pdf",  default=0,       type=float, action='store', help="Learning rate (pdf model)")

    parser.add_argument("-sr",       "--soft_reset",         default=0,       type=int,   action='store', help="Reset optimizer (can be useful if changing model architecture)")
    parser.add_argument("-ec",       "--empty_cache",        default=1,       type=int,   action='store', help="Empty cache (can be needed if VRAM limited)")
    
    parser.add_argument("-wd_w",     "--weight_decay_w",     default=1e-5,    type=float, action='store', help="Optimizer weight decay (note default for Adam 0.0 and AdamW 1e-2)!")
    parser.add_argument("-wd_b",     "--weight_decay_b",     default=0.0,     type=float, action='store', help="Optimizer weight decay (note default for Adam 0.0 and AdamW 1e-2)!")
    
    parser.add_argument("-wd_pdf_w", "--weight_decay_pdf_w", default=1e-5,    type=float, action='store', help="Optimizer weight decay (pdf model)")
    parser.add_argument("-wd_pdf_b", "--weight_decay_pdf_b", default=0.0,     type=float, action='store', help="Optimizer weight decay (pdf model)")
    
    parser.add_argument("-eps",      "--epsilon",      default=1e-6,           type=float, action='store', help="Optimizer epsilon (note default for AdamW 1e-8)!")
    parser.add_argument("-eps_pdf",  "--epsilon_pdf",  default=1e-6,           type=float, action='store', help="Optimizer epsilon (pdf model)")
    
    parser.add_argument("-mg",  "--max_grad_norm",     default=1.0,            type=float, action='store', help="Maximum gradient norm clipping value")
    parser.add_argument("-b",   "--batch_size",        default=1,              type=int,   action='store', help="Training batch size (num. of events between gradient update, increase within memory limits)")
    
    parser.add_argument("-st",  "--scheduler_type",    default='warm-cos',     type=str,   action='store', help="Scheduler type ('constant', 'cos', 'warm-cos', 'exp', 'step')")
    parser.add_argument("-sp",  "--scheduler_period",  default=10000,          type=int,   action='store', help="Learning rate scheduler period")
    parser.add_argument("-a",   "--auc_threshold",     default=0.998,          type=float, action='store', help="Pre-training edge-predictor AUC threshold")
    
    parser.add_argument("-e",   "--epoch",             default=0,              type=int,   action='store', help="Continue training from a previously saved iteration number (set -1 for the last timestamp)")
    parser.add_argument("-s",   "--save_period",       default=1200,           type=int,   action='store', help="Save model period (seconds)")
    
    parser.add_argument("-ft",  "--transfer_tag",      default=None,           type=str,   action='store', help="Transfer learning starting checkpoint name")
    parser.add_argument("-t",   "--save_tag",          default='default',      type=str,   action='store', help="Model checkpoint savename")
    
    parser.add_argument("-e0",  "--event_start",       default=[2450, 5950],  type=int,   nargs='+',      help="Training event file number (start)")
    parser.add_argument("-e1",  "--event_end",         default=[5899, 7499],  type=int,   nargs='+',      help="Training event file number (end)")
    
    parser.add_argument("-ev0", "--event_val_start",   default=[7500],        type=int,   nargs='+',      help="Validate event file number (start)")
    parser.add_argument("-ev1", "--event_val_end",     default=[7699],        type=int,   nargs='+',      help="Validate event file number (end)")
    
    parser.add_argument("-v",   "--validate",          default=0,             type=int,   action="store", help="Validate {0,1}")
    parser.add_argument("-r",   "--period",            default=1.0,           type=float, action='store', help="Input stochastic event switch probability (allows gradient descent to stay on the same event) [0,1] (1 = change every iteration)")
    
    parser.add_argument("-f0",  "--rfactor_start",     default=0.01,          type=float, action="store", help="TrackML re-sample: pile-up fraction (0,1] range start (1 = no reduction)")
    parser.add_argument("-f1",  "--rfactor_end",       default=0.01,          type=float, action="store", help="TrackML re-sample: pile-up fraction (0,1] range end   (1 = no reduction)")
    parser.add_argument("-nr",  "--noise_ratio",       default=0.05,          type=float, action="store", help="TrackML dataset re-sample: Noise ratio [0,1] (-1 = for original)")
    
    parser.add_argument("-n",   "--node2node",         default='hyper',       type=str,   action="store", help="TrackML dataset re-sample: Ground truth adjacency matrix type 'hyper' or 'mst'")
    parser.add_argument("-nc",  "--ncell",             default=131072,        type=int,   action="store", help="Number of cells")
    
    parser.add_argument("-fp",  "--fp_dtype",          default='float32',     type=str,   action="store", help="float32, bfloat16, float16")
    parser.add_argument("-d",   "--device",            default='auto',        type=str,   action="store", help="Device (auto, cpu, cuda)")
    parser.add_argument("-de",  "--debug",             default=0,             type=int,   action="store", help="Debug {0,1}")
    
    parser.add_argument("-m",   "--param",             default='tune-5',      type=str,   action="store", help="Parameter module name")
    parser.add_argument("-c",   "--cluster",           default='transformer', type=str,   action='store', help="Clustering algorithm ('transformer', 'cut', 'dbscan', 'hdbscan' or 'none')")
    parser.add_argument("-et",  "--edge_threshold",    default=0.55,          type=float, action='store', help="Graph edge cut threshold")
    
    parser.add_argument("-rng", "--random_seed",       default=12345,         type=int,   action='store', help="Random seed")
    
    args   = parser.parse_args()
    if args.device == 'auto':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    print(args)
    
    # Set random seeds
    torchtools.set_random_seed(seed=args.random_seed)
    
    # Load parameter module and models description module
    global_param = importlib.import_module(f'hypertrack.models.global_{args.param}')
    models       = importlib.import_module(f'hypertrack.models.models_{args.param}')
    
    # Create folders
    for folder in [f'{CWD}/figs/train/{args.save_tag}/metric', f'{CWD}/figs/train/{args.save_tag}/gradflow']:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # -----------------------------------------------------------
    # Models

    model = {
        'net': models.InverterNet(**global_param.net_model_param),
        'pdf': models.TrackFlowNet(**global_param.pdf_model_param)
    }
    
    if 'cuda' in args.device.type and torch.cuda.device_count() > 1: # Untested
        for key in model:
            model[key] = torch.nn.parallel.DistributedDataParallel(model[key])
    
    for key in model:
        model[key] = model[key].to(args.device)
    
    # Compile models (not here, only in the inference code)
    #for key in model:
    #    model[key] = torch.compile(model[key], mode="reduce-overhead")
    
    model['net'].training_on = True if args.learning_rate     > 0 else False
    model['pdf'].training_on = True if args.learning_rate_pdf > 0 else False

    do_pdf = True if args.learning_rate_pdf > 0 else False
        
    # -----------------------------------------------------------
    # Training multi-phase control
    
    # Set clustering algorithm
    global_param.cluster_param['algorithm']      = args.cluster
    global_param.cluster_param['edge_threshold'] = args.edge_threshold
    
    do_clustering = False

    model['net'].set_model_param_grad(string_id='edx', requires_grad=True)
    model['net'].set_model_param_grad(string_id='ccx', requires_grad=False) # Transformer not on by default

    # -----------------------------------------------------------
    ## Optimizers
    
    optimizer = auxtrain.set_optimizer(model=model, learning_rate=args.learning_rate, args=args)

    # -----------------------------------------------------------
    ## Schedulers
    
    scheduler = auxtrain.set_scheduler(optimizer=optimizer, args=args)

    # -----------------------------------------------------------
    ## Load voxdyn predictor object once into the memory
    try:
        with open(f"{CWD}/models/voxdyn_node2node_{args.node2node}_ncell_{args.ncell}.pkl", "rb") as input_file:
            print('Loading voxdyn data ...')
            geometric_obj = pickle.load(input_file)
            
            sparsity = np.sum(geometric_obj['C']) / float(geometric_obj['ncell']**2)
            print(f"Loaded voxdyn object: ncell = {geometric_obj['ncell']} | sparsity(C) = {sparsity:0.4f}")
            
            # Convert to csr-library sparse matrix (supports Numba)
            geometric_obj['C']  = csr.CSR.from_scipy(geometric_obj['C'])
            #geometric_obj['C3'] = csr.CSR.from_scipy(geometric_obj['C3'])
    
    except Exception as e:
        cprint(e, 'red')
        exit()
    
    # -----------------------------------------------------------
    # Load previous saved model states

    metrics = {
        'epoch': [],
        'learning_rate_net': [],
        'learning_rate_pdf': [],
        
        'trn_edge_auc': [],
        'val_edge_auc': [],
        'trn_majority_score': [],
        'val_majority_score': [],
        
        'trn_edge_BCE': [],
        'trn_edge_contrastive': [],
        'trn_cluster_BCE': [],
        'trn_cluster_contrastive': [],
        'trn_track_neglogpdf': [],
        'trn_cluster_neglogpdf': []
    }

    start_epoch = 0
    
    if args.epoch != 0:
        
        # Models
        try:
            metrics, found_epoch = auxtrain.load_models(
                epoch=args.epoch, model=model, args=args, transfer_tag=args.transfer_tag)
            
            for m in scheduler.keys():
                name = f'learning_rate_{m}'
                if name not in metrics:
                    metrics[name] = [0.0] * len(metrics['epoch'])
            
            start_epoch = found_epoch + 1
        
        except Exception as e:
            cprint(e, 'red')
        
        # Load Optimizers (unless we soft reset their parameters)
        if not args.soft_reset:
            try:
                auxtrain.load_optimizers(
                    epoch=args.epoch, optimizer=optimizer, scheduler=scheduler, args=args, transfer_tag=args.transfer_tag)
            
            except Exception as e:
                cprint(e, 'red')
                cprint('Starting training with default optimizer values', 'red')        
        else:
            cprint('Reset of training parameters to default', 'red')
            # Nothing more here, the default parameters loaded already above ^
        
        # -------------------------------------------------------
        # Check do we need the transformer training turned on
        
        if args.cluster == 'transformer' and (len(metrics['trn_majority_score']) > 0 and metrics['trn_majority_score'][-1] > 0):
            do_clustering = True
            model['net'].set_model_param_grad(string_id='edx', requires_grad=True)
            model['net'].set_model_param_grad(string_id='ccx', requires_grad=True)
            model['net'].training_on = True
        # -------------------------------------------------------
    
    for key in model.keys():
        torchtools.count_parameters_torch(model[key])
    
    # -----------------------------------------------------------
    # Set floating point precision
    
    if   args.fp_dtype == 'float32':
        dtype  = torch.float32
        scaler = None
    elif args.fp_dtype == 'bfloat16':
        dtype  = torch.bfloat16
        scaler = None
    elif args.fp_dtype == 'float16':
        dtype  = torch.float16
        scaler = {'net': torch.cuda.amp.GradScaler(), 'pdf': torch.cuda.amp.GradScaler()}
    else:
        raise Exception(__name__ + 'Unknown args.fp_dtype chosen')
    
    for key in model.keys():
        model[key] = model[key].to(dtype)
    
    if args.device.type != 'cpu':
        iotools.showmem_cuda(device=args.device)
    else:
        iotools.showmem()
    
    # -----------------------------------------------------------
    ## Iteration loop
    # (iteration ~ "epoch" (variable name) in this application
    # we do not keep count on standard epochs i.e. full iterations / loops over the whole sample)
    
    first         = True
    epoch         = start_epoch
    num_problems  = 0
    scheduler_sum = 0
    timer         = time.time()
    timer_start   = time.time()
    
    event_range          = iotools.construct_event_range(event_start=args.event_start, event_end=args.event_end)
    event_range_validate = iotools.construct_event_range(event_start=args.event_val_start, event_end=args.event_val_end)
    
    while True:
        
        if global_param.cluster_param['algorithm'] == 'none': # Force off
            do_clustering = False
        
        try:
            
            # ====================================================================           
            # Batch entry loop

            total_loss = {'net': 0.0, 'pdf': 0.0}
            loss       = {'edge_BCE': 0.0, 'edge_contrastive': 0.0, 'cluster_BCE': 0.0, 'cluster_contrastive': 0.0, 'track_neglogpdf': 0.0, 'cluster_neglogpdf': 0.0}
            scalar     = {'edge_auc': 0.0, 'majority_score': 0.0}

            tic_total  = time.time()
            
            for batch in range(args.batch_size):
                print(f'** batch: {batch+1} / {args.batch_size} **')
                
                if args.empty_cache: # Can be needed with large inputs
                    torch.cuda.empty_cache()
                    gc.collect()

                # Change input randomly
                if np.random.rand() < args.period or first:
                    event = random.choice(event_range)
                    first = False
                
                # Change pile-up mean (possibly within start,end range) to generalize the model
                rfactor = np.random.uniform(low=args.rfactor_start, high=args.rfactor_end)
                
                # Execute
                total_loss_this, loss_this, metric_this = \
                    auxtrain.execute( \
                        trainmode=True, model=model, event=event, rfactor=rfactor, \
                        do_clustering=do_clustering, do_pdf=do_pdf, args=args, global_param=global_param, geometric_obj=geometric_obj)

                ## Update losses
                for key in total_loss.keys():
                    total_loss[key] = total_loss[key] + total_loss_this[key] / args.batch_size
                for key in loss.keys():
                    loss[key]   = loss[key] + loss_this[key] / args.batch_size
                
                ## Update scalars
                for key in scalar.keys():
                    scalar[key] = scalar[key] + metric_this[key] / args.batch_size
            
            ## All events of the batch computed, update gradient
            auxtrain.backward(optimizer=optimizer, scheduler=scheduler, model=model, \
                total_loss=total_loss, scaler=scaler, max_grad_norm=args.max_grad_norm, dtype=dtype)
            
            # Add data for plots
            metrics['epoch'].append(epoch)
            
            for m in scheduler.keys():
                metrics[f'learning_rate_{m}'].append(float(scheduler[m].get_last_lr()[0]))
            
            for key in loss.keys():
                metrics[f'trn_{key}'].append(loss[key].item())

            for key in scalar.keys():
                metrics[f'trn_{key}'].append(scalar[key])
            
            # ====================================================================

            toc_0 = time.time() - tic_total
            
            # --------------------------------------------------------------------
            # VALIDATION
            tic = time.time()
            
            metric_this_val = {'edge_auc': 0.0, 'majority_score': 0.0}
            
            if args.validate:
                
                # Change input randomly
                if np.random.rand() < args.period:
                    event = random.choice(event_range_validate)

                # Change pile-up mean (possibly within start,end range) to generalize the model
                rfactor = np.random.uniform(low=args.rfactor_start, high=args.rfactor_end)
                
                # Execute
                total_loss_this, loss_this, metric_this_val = \
                    auxtrain.execute( \
                        trainmode=False, model=model, event=event, rfactor=rfactor, \
                        do_clustering=do_clustering, do_pdf=do_pdf, args=args, global_param=global_param, geometric_obj=geometric_obj)
            
            # Save scalar metrics
            for key in scalar.keys():
                try:
                    metrics[f'val_{key}'].append(metric_this_val[key])
                except:
                    True
            
            toc_1 = time.time() - tic

        except KeyboardInterrupt:
            cprint('You pressed ctrl+c', 'red')
            sys.exit()

        except Exception as e:
            
            cprint(f'{e}', 'red')
            print("-"*60)
            traceback.print_exc(file=sys.stdout)
            print("-"*60)
            cprint(f'Problem encountered in training, continue.', 'red')
            
            if num_problems >= 1E5:
                sys.exit()
            
            num_problems += 1
            
            continue
        
        # Clustering turned on (once), after the edge prediction is high enough
        if not do_clustering and metrics['trn_edge_auc'][-1] > args.auc_threshold:
            
            do_clustering = True
            
            if args.cluster == 'transformer':
                model['net'].set_model_param_grad(string_id='edx', requires_grad=True)
                model['net'].set_model_param_grad(string_id='ccx', requires_grad=True)
                model['net'].training_on = True
                torchtools.count_parameters_torch(model['net'])
                
                optimizer = auxtrain.set_optimizer(model=model, learning_rate=args.learning_rate, args=args)
                scheduler = auxtrain.set_scheduler(optimizer=optimizer, args=args)
        
        cprint(f'---------------', 'yellow')
        cprint(iotools.sysinfo(),  'yellow')
        cprint(f"[{args.param} | {args.save_tag}] Iteration: {epoch:03d} ({(time.time() - timer_start)/60:0.1f} min since [re]start) | Edge Train AUC: {metrics['trn_edge_auc'][-1]:0.4f}, Val AUC: {metrics['val_edge_auc'][-1]:0.4f}  [train: {toc_0:0.2f}: validate: {toc_1:0.2f} sec]", 'yellow')
        cprint(f'---------------', 'yellow')
        print('')
        
        if args.debug:
            
            for key in model.keys():
                torchtools.count_parameters_torch(model[key])
                torchtools.plot_grad_flow(model[key].named_parameters(), param_types=[''])
        
        if time.time()-timer > args.save_period and epoch != 0:
            
            if args.device.type != 'cpu':
                iotools.showmem_cuda(device=args.device)
            else:
                iotools.showmem()
            
            # Plot gradient flow
            auxtrain.plot_gradient_flow(epoch=epoch, model=model, args=args)
            
            # Plot (vector) metrics
            visualize.ROC_plot(metrics=[metric_this['edge_metrics']], labels=[f'Edge GNN (iteration={epoch})'],
                title=f"node2node = '{args.node2node}'", filename=f'{CWD}/figs/train/{args.save_tag}/metric/edge_ROC')

            # Plot (scalar) metrics
            auxtrain.plot_metrics(epoch=epoch, metrics=metrics, model=model, args=args, train_param=global_param.train_param)
            
            # Save models
            auxtrain.save_models(epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler, metrics=metrics, args=args)
            
            timer = time.time()
            
        epoch += 1


if __name__ == '__main__':
    main()
