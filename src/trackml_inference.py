# HyperTrack neural model inference evaluation code
# 
# m.mieskolainen@imperial.ac.uk, 2023

import sys
sys.path.append("../hypertrack")

import csr
import gc
import torch
import importlib
from argparse import ArgumentParser

import logging
from datetime import datetime
import torch_geometric
import traceback
import pickle
import numpy as np
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numba
import os
import copy
from tqdm import tqdm
from termcolor import cprint
import os

from hypertrack import hyper, tools, torchtools, measures, visualize, trackml, iotools, voxdyn
from hypertrack.visualize import crf

def main():
    CWD = os.getcwd()
    numba.config.THREADING_LAYER = 'tbb' # requires tbb to be installed
    
    parser = ArgumentParser()
    parser.add_argument("-e0", "--event_start",    default=[8750], type=int,   nargs='+',      help="Event file start")
    parser.add_argument("-e1", "--event_end",      default=[9999], type=int,   nargs='+',      help="Event file end")
    parser.add_argument("-r",  "--repeat",         default=1,      type=int,   action='store', help="Number of repetitions per event")
    
    parser.add_argument("-e",  "--epoch",          default=-1,     type=int,   action='store', help="Load model from this previously saved iteration (set -1 for the last timestamp)")
    
    parser.add_argument("-pr", "--pre_only",       default=0,      type=int,   action='store', help="Rung geometric adjacency predictor only (test mode)")
    
    parser.add_argument("-h0", "--hits_min",       default=4,      type=int,   action='store', help="Minimum number of hits per track for the ground truth")
    parser.add_argument("-h1", "--hits_max",       default=23,     type=int,   action='store', help="Maximum number of hits per track (for plots)")
    
    parser.add_argument("-f",  "--rfactor",        default=0.01,   type=float, action="store", help="TrackML dataset re-sample: pile-up fraction (0,1] (1 = no reduction)")
    parser.add_argument("-nr", "--noise_ratio",    default=0.05,   type=float, action="store", help="TrackML dataset re-sample: Noise ratio [0,1] (set -1 for the original)")
    
    parser.add_argument("-n",  "--node2node",      default='hyper',        type=str,   action="store", help="TrackML dataset re-sample: Ground truth adjacency matrix type")
    parser.add_argument("-nc", "--ncell",          default=131072,         type=int,   action="store", help="Number of cells")
    
    parser.add_argument("-fp", "--fp_dtype",       default='float32',      type=str,   action="store", help="float32, bfloat16, float16")
    parser.add_argument("-cm", "--compile",        default=1,              type=int,   action='store', help="Compile the neural model {0,1}")
    parser.add_argument("-mm", "--mode",           default='eval',         type=str,   action="store", help="Torch mode ('eval' or 'train') (use 'train' only for debugging)")
    
    parser.add_argument("-d",  "--device",         default='auto',         type=str,   action="store", help="Device (auto, cpu, cuda)")
    parser.add_argument("-t",  "--read_tag",       default='default',      type=str,   action='store', help="Input model checkpoint name")
    parser.add_argument("-o",  "--out_tag",        default='default',      type=str,   action='store', help="Output folder name")
    
    parser.add_argument("-m",  "--param",          default='tune-5',       type=str,   action="store", help="Parameter module name")
    parser.add_argument("-c",  "--cluster",        default='transformer',  type=str,   action='store', help="Clustering algorithm ('transformer', 'cut', 'dbscan', 'hdbscan')")
    parser.add_argument("-et", "--edge_threshold", default=0.55,           type=float, action='store', help="Edge cut threshold")
    
    parser.add_argument("-rng", "--random_seed",   default=12345,          type=int,   action='store', help="Random seed")

    args = parser.parse_args()
    
    # Set random seeds
    torchtools.set_random_seed(seed=args.random_seed)
    
    # Load parameter module and models description module
    global_param = importlib.import_module(f'hypertrack.models.global_{args.param}')
    models       = importlib.import_module(f'hypertrack.models.models_{args.param}')
    
    # Set clustering algorithm
    global_param.cluster_param['algorithm']      = args.cluster
    global_param.cluster_param['edge_threshold'] = args.edge_threshold
    
    if args.pre_only:
        global_param.geom_param['verbose'] = True

    # Output
    OUT_PATH = f'{CWD}/figs/inference/{args.out_tag}'
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    
    # -----------------------------------------------------------
    ## Load VD-estimator object once into the memory
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
    
    if args.device == 'auto':
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
    
    if   args.fp_dtype == 'float32':
        dtype  = torch.float32
    elif args.fp_dtype == 'bfloat16':
        dtype  = torch.bfloat16
    elif args.fp_dtype == 'float16':
        dtype  = torch.float16
    else:
        raise Exception(__name__ + 'Unknown args.fp_dtype chosen')
    
    # -----------------------------------------------------------
    # Load HyperTrack network
    if not args.pre_only:
        model = {
            'net': models.InverterNet(**global_param.net_model_param),
            'pdf': models.TrackFlowNet(**global_param.pdf_model_param)
        }
        
        if 'cuda' in args.device.type and torch.cuda.device_count() > 1:
            for key in model:
                model[key] = torch.nn.DistributedDataParallel(model[key])
        
        for key in model:
            model[key] = model[key].to(args.device)
        
        # Set floating point precision
        for key in model:
            model[key] = model[key].to(dtype=dtype)
        
        # Load models
        model, found_epoch = iotools.load_models(keys=['net', 'pdf'], model=model, epoch=args.epoch,
                                    read_tag=args.read_tag, device=args.device, mode=args.mode)
        
        # Compile models for maximal speed
        if args.compile:
            for key in model:
                model[key] = torch.compile(model[key], mode="reduce-overhead")
    
        for key in model.keys():
            torchtools.count_parameters_torch(model[key])
    
    # -----------------------------------------------------------
    # Event loop
    
    H_mix  = np.zeros((args.hits_max, args.hits_max), dtype=np.float32)   
    
    all_num_nodes    = np.array([], dtype=np.int32)
    all_num_edges    = np.array([], dtype=np.int32)
    all_time_VD      = np.array([], dtype=np.float32)
    all_time_GNN     = np.array([], dtype=np.float32)
    all_time_CC      = np.array([], dtype=np.float32)
    all_time_cluster = np.array([], dtype=np.float32)
    
    all_VD_tpr       = np.array([], dtype=np.float32)
    all_VD_fpr       = np.array([], dtype=np.float32)
    all_GNN_tpr      = np.array([], dtype=np.float32)
    all_GNN_fpr      = np.array([], dtype=np.float32)
    all_GNN_AUC      = np.array([], dtype=np.float32)
    
    all_DMS_scores   = np.array([], dtype=np.float32)
    all_DMS_eff      = np.array([], dtype=np.float32)
    all_DMS_purity   = np.array([], dtype=np.float32)
    
    all_passed       = None
    all_hit_purity   = None
    all_hit_eff      = None
    valid_events     = 0
    first_event      = True
    
    event_range = iotools.construct_event_range(event_start=args.event_start, event_end=args.event_end)
    
    for r in range(args.repeat):
        for k in tqdm(range(len(event_range))):
            
            event_number = event_range[k]

            torch.cuda.empty_cache() # Important
            gc.collect()
            
            try:
                print(f'Inference with dtype = {dtype}')
                
                # ------------------------------------------------------------------------
                # Step 0. Load event data
                
                X, x_ind, objects, hit_weights = \
                    trackml.load_reduce_event(event=event_number, rfactor=args.rfactor, noise_ratio=args.noise_ratio)

                # Compute the ground truth adjacency for metrics
                edge_index = tools.compute_ground_truth_A(X=X, x_ind=x_ind, node2node=args.node2node)
                A = tools.edges2adj(edge_index=edge_index, N=X.shape[0])

                # ------------------------------------------------------------------------
                # Step 1. Run geometrical pre-processor
                
                if   global_param.geom_param['algorithm'] == 'voxdyn':
                    edge_index_hat_np, A_hat, time_voxdyn   = voxdyn.voxdyn_predictor(X=X, obj=geometric_obj, device=global_param.geom_param['device'])
                elif global_param.geom_param['algorithm'] == 'neurodyn':
                    raise Exception('Neurodyn not implemented.')
                    #edge_index_hat_np, A_hat, time_neurodyn = neurodyn.neural_predictor(X=X, obj=geometric_obj, **global_param.geom_param)
                else:
                    raise Exception('Unknown geometrical algorithm chosen.')
                
                if global_param.geom_param['verbose']:
                    measures.adjgraph_metrics(A=A, A_hat=A_hat)

                if args.pre_only:
                    continue
                
                # Ground truth labels for each estimated edge (for GNN metrics)
                edge_label     = tools.create_edge_label(edge_index_hat=edge_index_hat_np, A=A)
                
                # Change datatypes
                X              = torch.tensor(X, dtype=dtype, device=args.device)
                edge_index_hat = torch.tensor(edge_index_hat_np, dtype=torch.long, device=args.device)
                objects_torch  = torch.tensor(objects, dtype=dtype, device=args.device)
                edge_label     = torch.tensor(edge_label, dtype=torch.long, device=args.device)
                
                # Feature normalization
                if global_param.normalize_input:
                    print(__name__ + f'.normalizing input X')
                    global_param.feature_scaler(X)
                
                # ** SORT (needed by certain message passing aggregators) **
                cprint(__name__ + '.execute: Sorting edge indices', 'red')
                edge_index_hat, edge_label = torch_geometric.utils.sort_edge_index(edge_index=edge_index_hat, edge_attr=edge_label, sort_by_row=False)
                
                # ------------------------------------------------------------------------
                # Step 2. Run the hypertrack network

                # Run the network
                time_neural = time.time()
                
                #with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16): # [EXPERIMENTAL]
                
                with torch.no_grad():
                        cluster_labels_hat, loss, aux = \
                            hyper.hypertrack(
                                model           = model,
                                X               = X,
                                edge_index_hat  = edge_index_hat,
                                x_ind           = None,
                                hit_weights     = None,
                                edge_label      = None,
                                true_clusters   = None,
                                objects         = None,
                                do_pdf          = False,
                                do_clustering   = True,
                                trainmode       = False,
                                cluster_param   = global_param.cluster_param,
                                train_param     = None,
                                device          = args.device)

                time_neural = time.time() - time_neural
                
                if args.device.type != 'cpu':
                    iotools.showmem_cuda(device=args.device)
                else:
                    iotools.showmem()
                
                # Transform clustering result format
                x_ind_hat = tools.create_cluster_ind(cluster_labels_hat.cpu().numpy(), hmax=args.hits_max)
                
                # ------------------------------------------------------------------------
                # Collect latencies (sec)
                
                time_VD      = time_voxdyn['time_L2'] + time_voxdyn['time_A']
                time_GNN     = aux['time_GNN']
                time_CC      = aux['time_CC']
                time_cluster = aux['time_cluster']

                # ------------------------------------------------------------------------
                # Performance comparisons
                
                # Voxel-Dynamic graph comparison
                gmetrics = measures.adjgraph_metrics(A=A, A_hat=A_hat, print_on=True if first_event else False)
                VD_tpr   = gmetrics['tpr']
                VD_fpr   = gmetrics['fpr']
                
                # ROC, AUC
                y_true   = edge_label.cpu().to(torch.float32).numpy()
                y_pred   = aux['edge_prob']
                met      = measures.Metric(y_true=y_true, y_pred=y_pred, roc_points=512)
                
                # Find cut threshold point performance
                index    = np.argmin(np.abs(met.thresholds - args.edge_threshold))
                GNN_tpr  = met.tpr[index]
                GNN_fpr  = met.fpr[index]
                GNN_AUC  = met.auc
                
                # For the first event
                if first_event:
                    
                    tools.print_clusters(x_ind=x_ind_hat)
                    
                    ## Input data distributions
                    X_np   = X.to(torch.float32).cpu().numpy()
                    fig,ax = plt.subplots()
                    plt.hist(X_np[:,0], bins=100, color=visualize.imperial_light_blue, histtype='step', label=crf('$x$', X_np[:,0]))
                    plt.hist(X_np[:,1], bins=100, color=visualize.imperial_green,      histtype='step', label=crf('$y$', X_np[:,1]))
                    plt.hist(X_np[:,2], bins=100, color=visualize.imperial_dark_red,   histtype='step', label=crf('$z$', X_np[:,2]))
                    plt.legend()
                    for f in ['pdf', 'png']:
                        plt.savefig(f'{OUT_PATH}/data_distributions.{f}', bbox_inches='tight')
                    plt.close(fig)
                    
                    ## Truth edges
                    y_true = tools.create_edge_label(edge_index_hat=edge_index_hat_np, A=A)
                    y_pred = aux['edge_prob']

                    self_ind    = (edge_index_hat_np[0,:] == edge_index_hat_np[1,:])
                    nonself_ind = (edge_index_hat_np[0,:] != edge_index_hat_np[1,:])
                    
                    visualize.density_MVA_wclass(y_pred=y_pred[self_ind], y=y_true[self_ind],
                                                label='$HyperTrack$ (GNN: self edges)', weights=None,
                                                num_classes=2, hist_edges=100, filename=f'{OUT_PATH}/predict_edge_score_self')

                    visualize.density_MVA_wclass(y_pred=y_pred[nonself_ind], y=y_true[nonself_ind],
                                                label='$HyperTrack$ (GNN: non-self edges)', weights=None,
                                                num_classes=2, hist_edges=100, filename=f'{OUT_PATH}/predict_edge_score_nonself')
                
                # Count number of clusters
                multiplicity             = tools.count_reco_cluster(x_ind,     hits_min=args.hits_min)
                multiplicity_hat         = tools.count_reco_cluster(x_ind_hat, hits_min=args.hits_min)            
                
                # Cluster hit multiplicity
                counts_hat, overflow_hat = tools.count_hit_multiplicity(x_ind_hat, bins=args.hits_max)
                counts, overflow         = tools.count_hit_multiplicity(x_ind,     bins=args.hits_max)
                
                ## Majority score
                DMS_score, DMS_eff, DMS_purity, passed, match_index = measures.majority_score(x_ind=x_ind, x_ind_hat=x_ind_hat, hit_weights=hit_weights, hits_min=args.hits_min)
                
                # Count hit mixing, efficiency, purity
                H_mix__, hit_eff, hit_purity = tools.compute_hit_mixing(x_ind=x_ind, x_ind_hat=x_ind_hat, match_index=match_index, hits_max=args.hits_max)
                H_mix += H_mix__
                
                # ------------------------------------------------------------------------
                cprint(iotools.sysinfo(), 'yellow')
                
                n_nonclustered = int(torch.sum(cluster_labels_hat == -1))
                
                print(f'Clustering DMS score = {DMS_score:0.3f} | Efficiency: {DMS_eff:0.3f} | Purity: {DMS_purity:0.3f} | Non-clustered hits = {n_nonclustered / len(cluster_labels_hat):0.3f}')
                cprint(f'VD step: {time_VD:0.2f} sec | GNN step: {time_GNN:0.2f} sec | CC step: {time_CC:0.2f} sec | Cluster step: {time_cluster:0.2f} sec | Total: {time_VD+time_GNN+time_CC+time_cluster:0.2f} sec', 'yellow')
                
                # Skip the first event due to JIT-compilation latency
                if first_event:
                    first_event = False
                    continue
                
                ## ------------------------------------------------------------------------
                # Collect summary data
                
                all_num_nodes    = np.append(all_num_nodes,    X.shape[0])
                all_num_edges    = np.append(all_num_edges,    edge_index_hat.shape[1])
                all_time_VD      = np.append(all_time_VD,      time_VD)
                all_time_GNN     = np.append(all_time_GNN,     time_GNN)
                all_time_CC      = np.append(all_time_CC,      time_CC)
                all_time_cluster = np.append(all_time_cluster, time_cluster)

                all_VD_tpr       = np.append(all_VD_tpr,     VD_tpr)
                all_VD_fpr       = np.append(all_VD_fpr,     VD_fpr)                
                all_GNN_tpr      = np.append(all_GNN_tpr,    GNN_tpr)
                all_GNN_fpr      = np.append(all_GNN_fpr,    GNN_fpr)
                all_GNN_AUC      = np.append(all_GNN_AUC,    GNN_AUC)
                all_DMS_scores   = np.append(all_DMS_scores, DMS_score)
                all_DMS_eff      = np.append(all_DMS_eff,    DMS_eff)
                all_DMS_purity   = np.append(all_DMS_purity, DMS_purity)
                
                # Collect event data
                if valid_events == 0:
                    all_passed           = copy.deepcopy(passed)
                    all_hit_purity       = copy.deepcopy(hit_purity)
                    all_hit_eff          = copy.deepcopy(hit_eff)
                    
                    all_objects          = copy.deepcopy(objects)
                    
                    all_multiplicity     = copy.deepcopy(multiplicity)
                    all_multiplicity_hat = copy.deepcopy(multiplicity_hat)
                    
                    all_counts           = copy.deepcopy(counts)
                    all_counts_hat       = copy.deepcopy(counts_hat)
                    
                else:
                    all_passed           = np.hstack((all_passed, passed))
                    all_hit_purity       = np.hstack((all_hit_purity, hit_purity))
                    all_hit_eff          = np.hstack((all_hit_eff, hit_eff))
                    
                    all_objects          = np.vstack((all_objects, objects))

                    all_multiplicity     = np.hstack((all_multiplicity, multiplicity))
                    all_multiplicity_hat = np.hstack((all_multiplicity_hat, multiplicity_hat))
                    
                    all_counts          += counts
                    all_counts_hat      += counts_hat
                
                valid_events += 1
            
            except KeyboardInterrupt:
                cprint('You pressed ctrl+c', 'red')
                sys.exit()

            except Exception as e:
                cprint(f'{e}', 'red')
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
                cprint(f'Problem encountered in inference, continue.', 'red')
                continue
    
    if args.pre_only:
        return
    
    logfile = f'{OUT_PATH}/results.log'
    logging.basicConfig(filename=logfile, level=logging.INFO, filemode='w', format='')
    
    # --------------------------------------------------------------------
    # SUMMARY TABLE
    
    def gs(xarr):
        x  = np.mean(xarr)
        dx = np.std(xarr) # We want to plot the standard deviation
        return tools.value_and_uncertainty_scientific(x=x, dx=dx)

    now       = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    
    logging.info(f'---------------------------')
    logging.info(f'Results ({dt_string})')
    logging.info(f'---------------------------')
    
    logging.info(f'Model tag:            {args.read_tag} ({found_epoch["net"]})')
    logging.info(f'Model parameters:     {args.param}')
    logging.info(f'')
    logging.info(f'Floating point type:  {args.fp_dtype}')
    logging.info(f'VD node-to-node:      {args.node2node}')
    logging.info(f'VD voxel count:       {args.ncell}')
    logging.info(f'GNN edge threshold:   {args.edge_threshold}')
    logging.info(f'Clustering algorithm: {args.cluster}')
    logging.info(f'')
    logging.info(f'Number of events:     {valid_events}')
    logging.info(f'Pile-up reduction:    {args.rfactor}')
    logging.info(f'Noise ratio:          {args.noise_ratio}')
    logging.info(f'')
    logging.info(f'')
    logging.info(f'Mean and (standard deviation)')
    logging.info(f'---------------------------')
    logging.info(f'VD(TPR) & VD(FPR) & GNN(TPR) & GNN(FPR) & GNN(AUC) & DMS & EFF & PUR & time_VD & time_GNN & time_CC & time_cluster \\\\')
    logging.info(f'')
    logging.info(f'{gs(all_VD_tpr)} & ${gs(all_VD_fpr)}$ & {gs(all_GNN_tpr)} & ${gs(all_GNN_fpr)}$ & {gs(all_GNN_AUC)} & {gs(all_DMS_scores)} & {gs(all_DMS_eff)} & {gs(all_DMS_purity)} & {gs(all_time_VD)} & {gs(all_time_GNN)} & {gs(all_time_CC)} & {gs(all_time_cluster)} \\\\')
    logging.info(f'---------------------------')

    # Print to the screen
    with open(logfile, 'r') as fin:
        print(fin.read())
    
    # --------------------------------------------------------------------
    # FIGURES

    ## Hit mixing matrix
    fig, ax = visualize.plot_mixing_matrix(X=H_mix, normalize='cols', cmap='coolwarm')
    
    plt.title('Number of hits (MC)', fontsize=9)
    plt.ylabel('Number of hits ($HyperTrack$)', fontsize=9)
    for f in ['pdf','png']:
        plt.savefig(f'{OUT_PATH}/hit_mixing.{f}', bbox_inches='tight')
    plt.close(fig)
    
    ## Num of edges, nodes vs double majority score scaling
    fig, ax = plt.subplots()
    plt.plot(all_num_nodes, all_DMS_scores, 'k.', label='Number of graph nodes')
    plt.plot(all_num_edges, all_DMS_scores, 'rs', label='Number of graph edges')
    plt.ylabel('Double majority score')
    plt.legend()
    ax.set_xscale('log')
    for f in ['pdf', 'png']:
        plt.savefig(f'{OUT_PATH}/predict_num_nodes_edges_vs_score.{f}', bbox_inches='tight')
    plt.close(fig)
    
    ## Num of nodes vs num of edges scaling
    fig, ax = plt.subplots()
    plt.plot(all_num_nodes, all_num_edges, 'k.', label='$HyperTrack$')
    plt.ylabel('Number of graph edges'); plt.xlabel('Number of graph nodes')
    plt.legend()
    ax.set_xscale('log')
    for f in ['pdf', 'png']:
        plt.savefig(f'{OUT_PATH}/predict_num_nodes_vs_num_edges.{f}', bbox_inches='tight')
    plt.close(fig)
    
    ## Estimator time scaling
    fig, ax = plt.subplots()
    plt.plot(all_num_nodes, all_time_VD,      '.', label=crf('VD',      all_time_VD))
    plt.plot(all_num_nodes, all_time_GNN,     '.', label=crf('GNN',     all_time_GNN))
    plt.plot(all_num_nodes, all_time_CC,      '.', label=crf('CC',      all_time_CC))
    plt.plot(all_num_nodes, all_time_cluster, '.', label=crf('Cluster', all_time_cluster))
    plt.plot(all_num_nodes, all_time_VD + all_time_GNN + all_time_CC + all_time_cluster, 'k.', label=crf('Total', all_time_VD+all_time_GNN+all_time_CC+all_time_cluster))
    
    plt.ylabel('Inference latency (s)')
    plt.xlabel('Number of graph nodes')
    plt.legend(loc='upper left')
    ax.set_xscale('log')
    for f in ['pdf', 'png']:
        plt.savefig(f'{OUT_PATH}/predict_num_nodes_vs_latency.{f}', bbox_inches='tight')
    plt.close(fig)
    
    ## Estimator time scaling
    fig, ax = plt.subplots()
    plt.plot(all_multiplicity, all_time_VD,      '.', label=crf('VD',      all_time_VD))
    plt.plot(all_multiplicity, all_time_GNN,     '.', label=crf('GNN',     all_time_GNN))
    plt.plot(all_multiplicity, all_time_CC,      '.', label=crf('CC',      all_time_CC))
    plt.plot(all_multiplicity, all_time_cluster, '.', label=crf('Cluster', all_time_cluster))
    plt.plot(all_multiplicity, all_time_VD + all_time_GNN + all_time_CC + all_time_cluster, 'k.', label=crf('Total', all_time_VD+all_time_GNN+all_time_CC+all_time_cluster))
    
    plt.ylabel('Inference latency (s)')
    plt.xlabel(f'MC clusters (nhits $\geq {args.hits_min}$)')
    plt.legend(loc='upper left')
    ax.set_xscale('log')
    for f in ['pdf', 'png']:
        plt.savefig(f'{OUT_PATH}/predict_clusters_vs_latency.{f}', bbox_inches='tight')
    plt.close(fig)
    
    ## Cluster count multiplicity histogram
    fig,ax = visualize.compare_cluster_multiplicity(multiplicity=all_multiplicity, \
        multiplicity_hat=all_multiplicity_hat, hits_min=args.hits_min)
    for f in ['pdf','png']:
        plt.savefig(f'{OUT_PATH}/predict_cluster_multiplicity.{f}', bbox_inches='tight')
    plt.close(fig)
    
    ## Cluster hit multiplicity histogram
    fig,ax = visualize.compare_hit_multiplicity(counts=all_counts, counts_hat=all_counts_hat)
    for f in ['pdf','png']:
        plt.savefig(f'{OUT_PATH}/predict_hit_multiplicity.{f}', bbox_inches='tight')
    plt.close(fig)
    
    ## Cluster metrics
    fig, ax = plt.subplots()
    plt.hist(all_DMS_scores, bins=np.linspace(0.5, 1.0, 100), color=visualize.imperial_dark_red,   histtype='step',
             label=crf('Cluster DMS', all_DMS_scores))
    plt.hist(all_DMS_eff, bins=np.linspace(0.5, 1.0, 100), color=visualize.imperial_green, histtype='step',
             label=crf('Cluster Efficiency', all_DMS_eff))
    plt.hist(all_DMS_purity, bins=np.linspace(0.5, 1.0, 100), color=visualize.imperial_orange, histtype='step',
             label=crf('Cluster Purity', all_DMS_purity))

    plt.legend(loc='upper left')
    plt.xlim(0.5, 1); plt.ylim(0, None)
    plt.ylabel('Event count'); plt.xlabel('Score')
    for f in ['pdf', 'png']:
        plt.savefig(f'{OUT_PATH}/predict_scores.{f}', bbox_inches='tight')
    plt.close(fig)

    ## Cluster hit efficiency and purity
    fig, ax = plt.subplots()
    plt.hist(all_hit_eff,    bins=np.linspace(0.5, 1.0, 100), color=visualize.imperial_dark_red,   histtype='step',
             label=crf('Cluster hit efficiency', all_hit_eff))
    plt.hist(all_hit_purity, bins=np.linspace(0.5, 1.0, 100), color=visualize.imperial_light_blue, histtype='step',
             label=crf('Cluster hit purity',     all_hit_purity))

    plt.legend(loc='upper left')
    plt.xlim(0.5, 1); plt.ylim(0.1, None)
    plt.yscale('log')
    plt.ylabel('Cluster count')
    for f in ['pdf', 'png']:
        plt.savefig(f'{OUT_PATH}/predict_cluster_hit_eff_purity.{f}', bbox_inches='tight')
    plt.close(fig)
    
    ## Efficiency histograms
    visualize.track_effiency_plots(passed=all_passed, purity=all_hit_purity, eff=all_hit_eff, objects=all_objects, hits_min=args.hits_min, path=OUT_PATH)

if __name__ == '__main__':
    main()

