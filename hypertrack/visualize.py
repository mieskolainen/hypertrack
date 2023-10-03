# Visualization tools
#
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from matplotlib.colors import LogNorm
from scipy.spatial import Voronoi, voronoi_plot_2d


from hypertrack import ratioplot
from hypertrack.tools import select_valid

from pylab import figure, cm

# Colors
imperial_dark_blue  = (0, 0.24, 0.45)
imperial_light_blue = (0, 0.43, 0.69)
imperial_dark_red   = (0.75, 0.10, 0.0)
imperial_green      = (0.0, 0.54, 0.23)
imperial_orange     = (0.82, 0.57, 0.57)
imperial_brown      = (0.80, 0.76, 0.57)

""" Global marker styles
zorder : approximate plotting order
lw     : linewidth
ls     : linestyle
"""
errorbar_style  = {'zorder': 3, 'ls': ' ', 'lw': 1, 'marker': 'o', 'markersize': 2.5}
plot_style      = {'zorder': 2, 'ls': '-', 'lw': 1}
hist_style_step = {'zorder': 0, 'ls': '-', 'lw': 1, 'histtype': 'step'}
hist_style_fill = {'zorder': 0, 'ls': '-', 'lw': 1, 'histtype': 'stepfilled'}
hist_style_bar  = {'zorder': 0, 'ls': '-', 'lw': 1, 'histtype': 'bar'}


def crf(label: str, values: np.ndarray):
    """
    Label string helper function
    
    Args:
        label:   arbitrary label
        values:  data array
    
    Returns:
        text string with "label (\mu  = x, \sigma = dx)"
    """
    return f'{label} ($\\mu = {np.mean(values):0.3f}, \\sigma = {np.std(values):0.3f}$)'


def plot_C_matrix(C: np.ndarray, title: str='', savename: str=''):
    """
    Visualize connectivity C-matrix
    
    Args:
        C:        connectivity matrix
        title:    plot title
        savename: plot savename
    
    Returns:
        plots saved directly to the disk
    """
    fig,ax = plt.subplots()
    CWD = os.getcwd()

    C = np.array(C, dtype=bool)
    C = ~C # Make 1 to white, 0 to black

    im = ax.matshow(C, cmap=cm.gray_r, norm=LogNorm(vmin=0.01, vmax=1))
    ax.set_title(title)
    
    for f in ['pdf', 'png']:
        plt.savefig(f'{CWD}/figs/voxdyn/C_matrix_{savename}.{f}', bbox_inches='tight')
    plt.close()


def plot_voronoi(centroids: np.ndarray, savename: str=''):
    """
    Visualize Voronoi division
    
    Args:
        centroids:  centroid vectors in an array [N x dim]
        savename:   name of the file to be saved (ending)
    
    Returns:
        plots saved directly to the disk
    """
    print(__name__ + f'.plot_voronoi: Running ...')
    CWD = os.getcwd()
    
    # xy
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    vor = Voronoi(centroids[:, [0,1]])
    fig = voronoi_plot_2d(vor, ax, show_vertices=False, line_colors='orange',
                line_width=2, line_alpha=0.8, point_size=3)
    plt.xlabel('$x$ (mm)')
    plt.ylabel('$y$ (mm)')
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.title(f'$N_{{cells}} = {len(centroids)}$')
    for f in ['pdf', 'png']:
        plt.savefig(f'{CWD}/figs/voxdyn/voronoi_xy_{savename}.{f}', bbox_inches='tight')
    plt.close()

    # zy
    fig, ax = plt.subplots(1,1, figsize=(16,32))
    vor = Voronoi(centroids[:, [2,1]])
    fig = voronoi_plot_2d(vor, ax, show_vertices=False, line_colors='orange',
                line_width=2, line_alpha=0.8, point_size=3)
    plt.xlabel('$z$ (mm)')
    plt.ylabel('$y$ (mm)')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title(f'$N_{{cells}} = {len(centroids)}$')
    for f in ['pdf', 'png']:
        plt.savefig(f'{CWD}/figs/voxdyn/voronoi_zy_{savename}.{f}', bbox_inches='tight')
    plt.close()

    # zx
    fig, ax = plt.subplots(1,1, figsize=(16,32))
    vor = Voronoi(centroids[:, [2,0]])
    fig = voronoi_plot_2d(vor, ax, show_vertices=False, line_colors='orange',
                line_width=2, line_alpha=0.8, point_size=3)
    plt.xlabel('$z$ (mm)')
    plt.ylabel('$x$ (mm)')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title(f'$N_{{cells}} = {len(centroids)}$')
    for f in ['pdf', 'png']:
        plt.savefig(f'{CWD}/figs/voxdyn/voronoi_zx_{savename}.{f}', bbox_inches='tight')
    plt.close()

def plot_tracks(X: np.ndarray, x_ind: np.ndarray, ind: list = [0],
                xbound=1250, zbound=3500, title='', text_on=True):
    """
    Visualize hits of tracks as 3D-points
    
    Args:
        X:      Hit position matrix [number of hits x 3]
        x_ind:  Hit indices of tracks array (-1 for null)
        ind:    Track indices to visualize
    
    Returns:
        fig, ax
    """
    
    if type(ind) is int: # Only one track
        ind = [ind]
    
    fig,ax = plt.subplots(2,1)
    fig.tight_layout()

    # xy
    plt.sca(ax[0])
    
    xyval = np.linspace(-xbound, xbound, 2)
    plt.plot(xyval, np.zeros(len(xyval)), 'k-', lw=1.0)
    plt.plot(np.zeros(len(xyval)), xyval, 'k-', lw=1.0)
    
    # Over all the tracks
    for i in ind:
        hit_index = select_valid(x_ind[i,:])
        
        # Over all the hits of the track
        for j in range(len(hit_index)):
            x  = X[hit_index[j],:]
            plt.plot(x[0], x[1], 'o', markersize=3)
            if text_on:
                plt.text(x[0], x[1], f'{j}', fontsize=5)

    ax[0].set_xlabel('$x$ (mm)')
    ax[0].set_ylabel('$y$ (mm)')
    ax[0].set_aspect('equal', 'box')
    ax[0].set_xlim(-xbound, xbound)
    ax[0].set_ylim(-xbound, xbound)
    ax[0].set_title(title)
    
    # zy
    plt.sca(ax[1])
    
    zval = np.linspace(-zbound, zbound, 2)
    plt.plot(zval, np.zeros(len(zval)), 'k-', lw=1.0)
    plt.plot(np.zeros(len(xyval)), xyval, 'k-', lw=1.0)
    
    # Over all the tracks
    for i in ind:
        hit_index = select_valid(x_ind[i,:])
        
        # Over all the hits of the track
        for j in range(len(hit_index)):
            x  = X[hit_index[j],:]
            plt.plot(x[2], x[1], 'o', markersize=3)
            if text_on:
                plt.text(x[2], x[1], f'{j}', fontsize=5)

    ax[1].set_xlabel('$z$ (mm)')
    ax[1].set_ylabel('$y$ (mm)')
    ax[1].set_aspect('equal', 'box')
    ax[1].set_ylim(-xbound, xbound)
    ax[1].set_xlim(-zbound, zbound)
    
    return fig, ax

def plot_adj_matrices(A:np.ndarray, A_hat:np.ndarray, maxrows:int=1000):
    """
    Visualize adjacency matrices

    Args:
        A:       Ground truth adjacency matrix
        A_hat:   Estimated adjacency matrix
        maxrows: Maximum number of rows to plot
    
    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(1,2)
    
    maxrows = np.min([A.shape[0], maxrows])

    ax[0].imshow(A[0:maxrows, 0:maxrows])
    ax[0].set_title('$A$')
    
    ax[1].imshow(A_hat[0:maxrows, 0:maxrows])
    ax[1].set_title('$\\hat{A}$')

    return fig,ax

def compare_cluster_multiplicity(multiplicity: np.ndarray, multiplicity_hat: np.ndarray,
                                 nbins: int=30, hits_min: int=1):
    """
    Compare ground truth and estimated cluster count multiplicity wise

    Args:
        multiplicity:     ground truth multiplicities
        multiplicity_hat: estimated multiplicities
        nbins:            number of histogram bins
    
    Returns:
        fig, ax
    """
    
    fig, ax = plt.subplots()
    
    counts, bins = np.histogram(multiplicity, nbins)
    plt.stairs(counts, bins, label=f'MC (nhits $\geq {hits_min}$)', color=(0.2,0.2,0.2))
    
    counts, bins = np.histogram(multiplicity_hat, bins)
    plt.stairs(counts, bins, label='$HyperTrack$', color=imperial_dark_red)

    plt.xlabel('$K$ (number of clusters)')
    plt.ylabel('Counts')
    plt.ylim([0, None])
    #plt.xlim([0, len(counts_hat)])
    plt.legend()
    
    # Set integer x-axis ticks
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return fig,ax

def compare_hit_multiplicity(counts, counts_hat):
    """
    Compare ground truth and estimated hit counts multiplicity wise

    Args:
        counts:     ground truth multiplicities
        counts_hat: estimated multiplicities
    
    Returns:
        fig, ax
    """
    
    print('Number of hits per cluster [estimate | true]:')

    for i in range(len(counts_hat)):
        print(f'[{i:2d}]:', end='')
        print(f'{counts_hat[i]:6d} ({counts_hat[i]/np.sum(counts_hat) * 100:5.2f} %) | ', end='')
        print(f'{counts[i]:6d} ({counts[i]/np.sum(counts) * 100:5.2f} %)')
    
    fig, ax = plt.subplots()
    plt.step(range(len(counts_hat)), counts,     label='MC (nhits > 0)', where='mid', color=(0.2,0.2,0.2))
    plt.step(range(len(counts_hat)), counts_hat, label='$HyperTrack$',   where='mid', color=imperial_dark_red)
    plt.xlabel('$N$ (hits per cluster)')
    plt.ylabel('Counts')
    plt.ylim([0, None])
    plt.xlim([0, len(counts_hat)])
    plt.legend()
    
    # Set integer x-axis ticks
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return fig,ax

def plot_mixing_matrix(X, normalize='cols', null_empty=True, cmap='coolwarm',
                       decimals=0, text_fontsize=6, color='w'):
    """
    Plot mixing matrix
    
    Args:
        X:             hit mixing matrix
        normalize:     normalize either 'cols' or 'rows' to sum to one
        null_empty:    set empty elements to NaN
        cmap:          colormap
        decimals:      number of decimal digits to include
        text_fontsize: fontsize
        color:         color
        
    Returns:
        fix, ax
    """
    
    # Normalize each column (row) to sum to 1 and scale to percents
    if   normalize == 'cols':
        summ = X.sum(axis=0,keepdims=1); ind = np.sum(summ, 0) > 0
        X[:,ind] = 100 * X[:,ind] / summ[:,ind]
    
    elif normalize == 'rows':
        summ = X.sum(axis=1,keepdims=1); ind = np.sum(summ, 1) > 0
        X[ind,:] = 100 * X[ind,:] / summ[ind,:]
    
    # Set empty
    if null_empty:
        X[X == 0] = np.nan
    
    fig, ax = plt.subplots()
    ax.matshow(X, cmap=mpl.colormaps[cmap])
    
    xlabels = [f'{i}' for i in range(X.shape[0])]
    ylabels = [f'{i}' for i in range(X.shape[0])]

    annotate_heatmap(X=X, ax=ax, xlabels=xlabels, ylabels=ylabels, decimals=decimals, color=color, fontsize=text_fontsize)

    return fig,ax


def annotate_heatmap(X, ax, xlabels, ylabels, x_rot = 0, y_rot = 0, decimals = 1, color = "w", fontsize=6):
    """ 
    Add text annotations to a matrix heatmap plot
    
    Args:
        X:         
        ax:        
        xlabels:   
        ylabels:   
        x_rot:     
        y_rot:     
        decimals:  
        color:     
        fontsize:  

    Returns:
        ax:        axis object
    """

    ax.set_xticks(np.arange(0, len(xlabels), 1));
    ax.set_yticks(np.arange(0, len(ylabels), 1));

    ax.set_xticklabels(labels=xlabels, rotation=x_rot, fontsize='xx-small')
    ax.set_yticklabels(labels=ylabels, rotation=y_rot, fontsize='xx-small')

    # Loop over data dimensions and create text annotations.
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):

            if (decimals == 0):
                text = ax.text(j, i, f'{X[i,j]:.0f}', ha="center", va="center", color=color, fontsize=fontsize)
            if (decimals == 1):
                text = ax.text(j, i, f'{X[i,j]:.1f}', ha="center", va="center", color=color, fontsize=fontsize)
            if (decimals == 2):
                text = ax.text(j, i, f'{X[i,j]:.2f}', ha="center", va="center", color=color, fontsize=fontsize)
    
    return ax


def hist_filled_error(ax, bins, y, err, color, **kwargs):
    """
    Stephist style error
    
    Args:
        ax:    axis object
        bins:  bins (central)
        y:     y-axis values
        err:   error bar values
        color: color value
        
    Returns:
        directly manipulates ax object
    """
    new_args = kwargs.copy()
    new_args['lw'] = 0
    new_args.pop('histtype', None) # Remove

    ax.fill_between(bins[0:-1], y-err, y+err, step='post', alpha=0.15, color=color, **new_args)

    # The last bin
    ax.fill_between(bins[-2:],  (y-err)[-2:], (y+err)[-2:], step='pre', alpha=0.15, color=color, **new_args)


def ratioerr(A: np.ndarray, B: np.ndarray, sigma_A: np.ndarray, sigma_B: np.ndarray,
             sigma_AB: np.ndarray = 0, EPS = 1E-15):
    """
    Ratio f(A,B) = A/B uncertainty, by Taylor expansion of the ratio
    
    Args:
        A:         values A
        B:         values B
        sigma_A:   1 sigma uncertainty of A
        sigma_B:   1 sigma uncertainty of B
        sigma_AB:  covariance between A and B (default 0)
    
    Returns:
        ratio uncertainty array
    """
    A[np.abs(A) < EPS] = EPS
    B[np.abs(B) < EPS] = EPS
    return np.abs(A/B) * np.sqrt((sigma_A/A)**2 + (sigma_B/B)**2 - 2*sigma_AB/(A*B))


def track_effiency_plots(passed: np.ndarray, purity: np.ndarray, eff: np.ndarray,
                         objects: np.ndarray, hits_min: int=4, path: str='figs/predict/'):
    """
    Create tracking efficiency and the hit set efficiency & purity plots
    
    Args:
        passed:    list of 1/0 indices indicating a double major score reconstructed particle
        purity:    clustering constituent purity for each ground truth reconstructed particle
        eff:       clustering constituent efficiency for each ground truth reconstructed particle
        
        objects:   particle (track) data with columns:
                   [3-vertex {1,2,3}, 3-momentum {4,5,6}, charge {7}, nhits {8}]
        hits_min:  minimum number of hits per ground truth (denominator definition)
        path:      figure save path
    
    Returns:
        figures saved to the disk
    """
    if len(passed) != len(objects):
        raise Exception(__name__ + f'.efficiency_plots: len(passed) != len(objects)')

    if objects.shape[1] != 8:
        raise Exception(__name__ + f'.efficiency_plots: "objects" should have 8 columns')

    # ==========================================================
    
    # As produced by trackml class
    vx_ind    = 0
    vy_ind    = 1
    vz_ind    = 2
    px_ind    = 3
    py_ind    = 4
    pz_ind    = 5
    Q_ind     = 6
    nhits_ind = 7

    def func_p3mod():
        return np.sqrt(objects[:, px_ind]**2 + objects[:, py_ind]**2 + objects[:, pz_ind]**2)

    def func_pt():
        return np.sqrt(objects[:, px_ind]**2 + objects[:, py_ind]**2)

    def func_px():
        return objects[:, px_ind]

    def func_py():
        return objects[:, py_ind]

    def func_pz():
        return objects[:, pz_ind]

    def func_r0mod():
        return np.sqrt(objects[:,vx_ind]**2 + objects[:,vy_ind]**2 + objects[:,vz_ind]**2)

    def func_r0xy():
        return np.sqrt(objects[:,vx_ind]**2 + objects[:,vy_ind]**2)

    def func_r0z():
        return np.abs(objects[:,vz_ind])
    
    def func_eta():
        theta    = np.arctan2(func_pt(), func_pz())
        cosTheta = np.cos(theta)
        return -0.5*np.log((1.0 - cosTheta) / (1.0 + cosTheta))
    
    def func_phi():
        return np.arctan2(func_py(), func_px())

    def func_nhits():
        return objects[:, nhits_ind]

    r0mod_obs = {
        'xlabel': '$|r^0|$ (mm)',
        'bins':   np.linspace(0.0, 50.0, 50),
        'xscale': 'linear',
        'x':      func_r0mod
    }
    
    r0xy_1_obs = {
        'xlabel': '$r_{xy}^0$ (mm)',
        'bins':   np.linspace(0.0, 0.1, 50),
        'xscale':  'linear',
        'x':      func_r0xy
    }
    r0xy_2_obs = {
        'xlabel': '$r_{xy}^0$ (mm)',
        'bins':   np.linspace(1.0, 10, 50),
        'xscale':  'linear',
        'x':      func_r0xy
    }
    r0xy_3_obs = {
        'xlabel': '$r_{xy}^0$ (mm)',
        'bins':   np.linspace(10, 100, 50),
        'xscale':  'linear',
        'x':      func_r0xy
    }
    r0xy_4_obs = {
        'xlabel': '$r_{xy}^0$ (mm)',
        'bins':   np.linspace(100, 1000, 50),
        'xscale':  'linear',
        'x':      func_r0xy
    }
    
    r0z_obs = {
        'xlabel': '$r_{z}^0$ (mm)',
        'bins':   np.linspace(0.0, 50.0, 50),
        'xscale':  'linear',
        'x':      func_r0z
    }
    p3mod_obs = {
        'xlabel': '$|p|$ (GeV)',
        'bins':   np.linspace(0.0, 3.0, 50),
        'xscale': 'linear',
        'x':      func_p3mod
    }
    pt_low_obs = {
        'xlabel': '$p_T$ (GeV)',
        'bins':   np.linspace(0.0, 3.0, 50),
        'xscale': 'linear',
        'x':      func_pt
    }
    pt_high_log_obs = {
        'xlabel': '$p_T$ (GeV)',
        'bins':   np.logspace(np.log10(0.1), np.log10(20.0), 50),
        'xscale': 'log',
        'x':      func_pt
    }
    eta_obs = {
        'xlabel': '$\\eta$',
        'bins':   np.linspace(-4.1, 4.1, 50),
        'xscale': 'linear',
        'x':      func_eta
    }
    phi_obs = {
        'xlabel': '$\\phi$ (rad)',
        'bins':   np.linspace(0, np.pi, 50),
        'xscale': 'linear',
        'x':      func_phi
    }
    nhits_obs = {
        'xlabel': 'nhits',
        'bins':   np.linspace(0.5, 25.5, 26),
        'xscale': 'linear',
        'x':      func_nhits
    }

    observables = {'r0mod':  r0mod_obs,
                   'r0xy_1': r0xy_1_obs,
                   'r0xy_2': r0xy_2_obs,
                   'r0xy_3': r0xy_3_obs,
                   'r0xy_4': r0xy_4_obs,
                   'r0z':    r0z_obs,
                   'p3mod':  p3mod_obs,
                   'pt_low':      pt_low_obs,
                   'pt_high_log': pt_high_log_obs,
                   'eta':    eta_obs,
                   'phi':    phi_obs,
                   'nhits':  nhits_obs}

    def compute_efficiency(A, B):
        """ 
        Efficiency ratio of counts A/B and a simple Taylor error propagation of the ratio 
        """
        R      = np.zeros(len(A))
        ind = B > 0
        R[ind] = A[ind] / B[ind]
        
        R_err  = ratioerr(A=A, B=B, sigma_A=np.sqrt(A), sigma_B=np.sqrt(B))
        return R, R_err
    
    # ==========================================================
    
    for key in observables.keys():

        print(__name__ + f'.effiency_plots: Computing observable <{key}>')

        x      = observables[key]['x']()
        bins   = observables[key]['bins']
        cbins  = (bins[:-1] + bins[1:]) / 2
        xlabel = observables[key]['xlabel']
        xscale = observables[key]['xscale']
        
        # Over different categories
        for category_type in ['inclusive', 'r0_xy_less_10mm', 'r0_xy_more_10mm']:
            
            if   category_type == 'inclusive':
                category_ind   = np.ones(objects.shape[0], dtype=bool)
                category_title = 'category: inclusive'
            
            elif category_type == 'r0_xy_less_10mm':
                category_ind   = func_r0xy() <= 10
                category_title = 'category: $r_{xy}^0 < 10$ mm'
            
            elif category_type == 'r0_xy_more_10mm':
                category_ind   = func_r0xy() > 10
                category_title = 'category: $r_{xy}^0 > 10$ mm'
            
            else:
                raise Exception(__name__ + f'.track_efficiency_plots: Unknown category_type: {category_type}')
            
            selected0       = np.logical_and((objects[:,nhits_ind] > 0), category_ind)
            selected        = np.logical_and((objects[:,nhits_ind] >= hits_min), category_ind)
            selected_passed = np.logical_and(selected, passed)

            # Create dual axes
            fig,ax = ratioplot.create_axes(figsize=(5.6, 4.8), \
                xlabel=xlabel, xlim=(np.min(bins),np.max(bins)), ylim=None, \
                ylim_ratio=(0.0, 1.1), ytick_ratio_step=0.1, height_ratios=(2.25, 1.0))
            
            # ----------------------------------
            ## Plot counts
            plt.sca(ax[0])
            
            count_MC0,_,_   = plt.hist(x[selected0],       bins, label=f'MC (nhits $> 0$)', histtype='step', color=(0.5,0.5,0.5))
            count_MC,_,_    = plt.hist(x[selected],        bins, label=f'MC (nhits $\geq {hits_min}$)', histtype='step', color=(0.2,0.2,0.2))
            count_hyper,_,_ = plt.hist(x[selected_passed], bins, label='$HyperTrack$', histtype='step', color=imperial_dark_red)
            
            plt.ylabel('Count', fontsize=10)
            plt.legend(fontsize=9, frameon=False)
            
            plt.xlim(bins[0], bins[-1])
            plt.ylim(0, None)
            plt.xscale(xscale)
            
            plt.title(category_title, fontsize=10)
            
            # ----------------------------------
            ## Plot ratios
            
            plt.sca(ax[1])
            ratioplot.plot_horizontal_line(ax[1], linestyle='--')
            
            ## Efficiency of HyperTrack wrt MC(nhits >= hits_min)
            e,e_err = compute_efficiency(A=count_hyper, B=count_MC)
            hist_filled_error(ax=ax[1], bins=bins, y=e, err=e_err, color=imperial_dark_red)
            plt.hist(x=cbins, bins=bins, weights=e, color=imperial_dark_red, label=f'Efficiency', **hist_style_step)
            
            ## Average cluster hit set purity and efficiency per kinematic bin
            hit_pur = np.zeros(len(cbins))
            hit_eff = np.zeros(len(cbins))
            for i in range(len(cbins)):
                
                ind  = np.logical_and(np.logical_and(bins[i] < x, x <= bins[i+1]), selected_passed)
                
                # Purity
                purity__ = purity[ind]
                hit_pur[i] = 0 if len(purity__) == 0 else np.mean(purity__)
                
                # Efficiency
                eff__    = eff[ind]
                hit_eff[i] = 0 if len(eff__) == 0 else np.mean(eff__)
            
            plt.hist(x=cbins, bins=bins, weights=hit_pur, color=imperial_green, label=f'Hit $\\langle$purity$\\rangle$',     **hist_style_step)
            plt.hist(x=cbins, bins=bins, weights=hit_eff, color=imperial_brown, label=f'Hit $\\langle$efficiency$\\rangle$', **hist_style_step)
            
            plt.xscale(xscale)
            
            plt.xlabel(xlabel, fontsize=10)
            plt.ylabel('')
            plt.legend(loc='lower right', fontsize=8)
            
            # Create directory
            OUT_PATH = f'{path}/{category_type}'
            if not os.path.exists(OUT_PATH):
                os.makedirs(OUT_PATH)
            
            for f in ['pdf', 'png']:
                plt.savefig(f'{OUT_PATH}/predict_obs_{key}.{f}', bbox_inches='tight')
            plt.close(fig)


def ROC_plot(metrics, labels, title = '', plot_thresholds=True,
    thr_points_signal = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
    filename = 'ROC', legend_fontsize=7, xmin=1e-5, alpha=0.32):
    """
    Receiver Operating Characteristics, i.e.,
    False Positive Rate (efficiency) vs True positive Rate (efficiency)
    
    Args:
        metrics:            metrics object
        labels:             labels
        title:              plot title
        plot_thresholds:    plot annotation points (True, False)
        thr_points_signal:  signal efficiency 
        filename:           filename (full path)
        legend_fontsize:    legend fontsize
        xmin:               minimum FPR value (x-axis)
        alpha:              transparency
    
    Returns:
        plots saved directly to the disk
    """
    
    for k in [0,1]: # linear & log
        
        fig,ax = plt.subplots()
        xx     = np.logspace(-5, 0, 100)
        plt.plot(xx, xx, linestyle='--', color='black', linewidth=1) # ROC diagonal

        for i in range(len(metrics)):

            if         i < 10:
                linestyle = '-'
            elif 10 <= i < 20:
                linestyle = '--'
            else:
                linestyle = ':'
            marker = 'None'
            
            if metrics[i] is None:
                print(__name__ + f'.ROC_plot: metrics[{i}] ({labels[i]}) is None, continue without')
                continue

            fpr        = metrics[i].fpr
            tpr        = metrics[i].tpr
            thresholds = metrics[i].thresholds

            if metrics[i].tpr_bootstrap is not None:

                # Percentile bootstrap based uncertainties
                tpr_lo = cortools.percentile_per_dim(x=metrics[i].tpr_bootstrap, q=100*(alpha/2))
                tpr_hi = cortools.percentile_per_dim(x=metrics[i].tpr_bootstrap, q=100*(1-alpha/2))
                
                fpr_lo = cortools.percentile_per_dim(x=metrics[i].fpr_bootstrap, q=100*(alpha/2))
                fpr_hi = cortools.percentile_per_dim(x=metrics[i].fpr_bootstrap, q=100*(1-alpha/2))

            # Autodetect a ROC point triangle (instead of a curve)
            if len(np.unique(fpr)) == 3:
                tpr = tpr[1:-1] # Remove first and last
                fpr = fpr[1:-1] # Remove first and last
                
                linestyle = "None"
                marker    = 'o'
            
            """
            # ROC-curve
            elif not (isinstance(fpr, int) or isinstance(fpr, float)):
                fpr    = fpr[1:] # Remove always the first element for log-plot reasons
                tpr    = tpr[1:]
                if metrics[i].tpr_bootstrap is not None:
                    tpr_lo = tpr_lo[1:]
                    tpr_hi = tpr_hi[1:]
                    fpr_lo = fpr_lo[1:]
                    fpr_hi = fpr_hi[1:]
            """
            
            ## Plot it
            plt.plot(fpr, tpr, drawstyle='steps-mid', color=f'C{i}', linestyle=linestyle, marker=marker, label = f'{labels[i]}: AUC = {metrics[i].auc:.4f}')
            
            # Uncertainty band
            if marker == 'None' and (metrics[i].tpr_bootstrap is not None):
                
                plt.fill_between(fpr,  tpr_lo, tpr_hi, step='mid', alpha=0.2, color=f'C{i}', edgecolor='none') # vertical
                plt.fill_betweenx(tpr, fpr_lo, fpr_hi, step='mid', alpha=0.2, color=f'C{i}', edgecolor='none') # horizontal

                # draw_error_band(ax=ax, x=fpr, y=tpr, \
                #   x_err=np.std(metrics[i].fpr_bootstrap, axis=0)[1:], \
                #   y_err=np.std(metrics[i].tpr_bootstrap, axis=0)[1:], \
                #   facecolor=f'C{i}', edgecolor="none", alpha=0.2)

            # Plot corresponding threshold points
            if plot_thresholds:
                
                try:
                    for eff in thr_points_signal:
                        index = np.argmin(np.abs(tpr - eff))
                        if fpr[index] >= xmin and fpr[index] <= 1.0:
                            plt.plot(fpr[index], tpr[index], '.', color=f'C{i}')
                            t = plt.text(x=fpr[index], y=tpr[index], s=f'{thresholds[index]:0.4g}', fontsize=5, color=f'C{i}')
                            t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                except: # If failed
                    True
        
        ax.set_xlabel('False Positive Rate $\\alpha$ (background efficiency)')
        ax.set_ylabel('True Positive Rate $1-\\beta$ (signal efficiency)')
        ax.set_title(title, fontsize=10)
        
        # Legend
        if len(metrics) > 12: # Put outside the figure
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize)
        else:
            plt.legend(loc='lower right', fontsize=legend_fontsize)
        
        if k == 0: # Linear-Linear

            plt.ylim(0.0, 1.0)
            plt.xlim(0.0, 1.0)
            plt.locator_params(axis="x", nbins=11)
            plt.locator_params(axis="y", nbins=11)

            ax.set_aspect(1.0 / ax.get_data_ratio() * 1.0)
            plt.savefig(filename + '.pdf', bbox_inches='tight')
        
        if k == 1: # Log-Linear

            plt.ylim(0.0, 1.0)
            plt.xlim(xmin, 1.0)
            plt.locator_params(axis="x", nbins=int(-np.log10(xmin) + 1))
            plt.locator_params(axis="y", nbins=11)

            plt.gca().set_xscale('log')
            ax.set_aspect(1.0 / ax.get_data_ratio() * 0.75)
            plt.savefig(filename + '--log.pdf', bbox_inches='tight')

        plt.close()


def density_MVA_wclass(y_pred: np.ndarray, y: np.ndarray, label: str, weights: np.ndarray=None,
                       num_classes: int=None, hist_edges: int=80, filename: str='output'):
    """
    Evaluate MVA output (1D) density per class
    
    Args:
        y_pred     :  MVA algorithm output
        y          :  Output (truth level target) data
        label      :  Label of the MVA model (string)
        weights    :  Sample weights
        hist_edges :  Histogram edges list (or number of bins, as an alternative)
    
    Returns:
        Plot pdf saved directly to the disk
    """
    
    # Make sure it is 1-dim array of length N (not N x num classes)
    if (weights is not None) and len(weights.shape) > 1:
        weights = np.sum(weights, axis=1)
    
    if weights is not None:
        classlegs = [f'$\\mathcal{{C}} = {k}$, $N={np.sum(y == k)}$ (weighted {np.sum(weights[y == k]):0.1f})' for k in range(num_classes)]
    else:
        classlegs = [f'$\\mathcal{{C}} = {k}$, $N={np.sum(y == k)}$ (no weights)' for k in range(num_classes)]

    # Over classes
    fig,ax = plt.subplots()
    
    for k in range(num_classes):
        ind = (y == k)

        w = weights[ind] if weights is not None else None
        x = y_pred[ind]

        hI, bins, patches = plt.hist(x=x, bins=np.linspace(0,1,hist_edges), weights=w,
            density = True, histtype = 'step', fill = False, linewidth = 2, label = 'inverse')
    
    plt.legend(classlegs, loc='upper center')
    plt.xlabel('MVA output $f(\\mathbf{{x}})$')
    plt.ylabel('density')
    plt.title(f'{label}', fontsize=9)
    
    for scale in ['linear', 'log']:
        ax.set_yscale(scale)
        savename  = f'{filename}--{scale}.pdf'
        plt.savefig(savename, bbox_inches='tight')
    
    # --------        
    fig.clf()
    plt.close()
