# Ratio plot helper functions
#
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np
import math
import matplotlib.pyplot as plt


def stepspace(start, stop, step):
    """ Linear binning edges between [start, stop]
    """
    return np.arange(start, stop + step, step)


def plot_horizontal_line(ax, color=(0.5,0.5,0.5), linewidth=0.9, ypos=1.0, linestyle='-'):
    """ For the ratio plot
    """
    xlim = ax.get_xlim()
    ax.plot(np.linspace(xlim[0], xlim[1], 2), np.array([ypos, ypos]), color=color, linewidth=linewidth, linestyle=linestyle)


def tick_calc(lim, step, N=6):
    """ Tick spacing calculator.
    """
    return [np.round(lim[0] + i*step, N) for i in range(1 + math.floor((lim[1] - lim[0])/step))]


def set_axis_ticks(ax, ticks, dim='x'):
    """ Set ticks of the axis.
    """
    if   (dim == 'x'):
        ax.set_xticks(ticks)
        ax.set_xticklabels(list(map(str, ticks)))
    elif (dim == 'y'):
        ax.set_yticks(ticks)
        ax.set_yticklabels(list(map(str, ticks)))


def tick_creator(ax, xtick_step=None, ytick_step=None, ylim_ratio=(0.7, 1.3),
        ratio_plot=True, minorticks_on=True, ytick_ratio_step=0.15, labelsize=9,
        labelsize_ratio=8, **kwargs) :
    """ Axis tick constructor.
    """

    # Get limits
    xlim = ax[0].get_xlim()
    ylim = ax[0].get_ylim()

    # X-axis
    if (xtick_step is not None):
        ticks = tick_calc(lim=xlim, step=xtick_step)
        set_axis_ticks(ax[-1], ticks, 'x')

    # Y-axis
    if (ytick_step is not None):    
        ticks = tick_calc(lim=ylim, step=ytick_step)
        set_axis_ticks(ax[0], ticks, 'y')

    # Y-ratio-axis
    if ratio_plot:
        ax[0].tick_params(labelbottom=False)
        ax[1].tick_params(axis='y', labelsize=labelsize_ratio)

        ticks = tick_calc(lim=ylim_ratio, step=ytick_ratio_step)
        ticks = ticks[:-1] # Remove the last
        set_axis_ticks(ax[1], ticks, 'y')
    
    # Tick settings
    for a in ax:
        if minorticks_on: a.minorticks_on()
        a.tick_params(top=True, bottom=True, right=True, left=True, which='both', direction='in', labelsize=labelsize)

    return ax

def create_axes(xlabel='$x$', ylabel=r'Counts', ylabel_ratio='Ratio',
    xlim=(0,1), ylim=None, ylim_ratio=(0.7, 1.3), height_ratios=(3.333, 1),
    ratio_plot=True, figsize=(5,4), fontsize=9, units={'x': '', 'y': ''}, density=False, **kwargs):
    """ Axes creator.
    """
    
    # Create subplots
    N = 2 if ratio_plot else 1
    gridspec_kw = {'height_ratios': height_ratios if ratio_plot else (1,), 'hspace': 0.0}
    fig, ax = plt.subplots(N,  figsize=figsize, gridspec_kw=gridspec_kw)
    ax = [ax] if (N == 1) else ax

    # Axes limits
    for a in ax:
        if xlim is not None:
            a.set_xlim(*xlim)

    if ylim is not None:
        ax[0].set_ylim(*ylim)

    # Axes labels
    if density:
        ylabel = f'$1/N$  {ylabel} / [{units["x"]}]'
    else:
        ylabel = f'{ylabel}  [{units["y"]} / {units["x"]}]'
    xlabel = f'{xlabel} [{units["x"]}]'
    
    ax[0].set_ylabel(ylabel, fontsize=fontsize)
    ax[-1].set_xlabel(xlabel, fontsize=fontsize)

    # Ratio plot
    if ratio_plot:
        ax[1].set_ylabel(ylabel_ratio, fontsize=fontsize)
        ax[1].set_ylim(*ylim_ratio)

    # Setup ticks
    ax = tick_creator(ax=ax, ratio_plot=ratio_plot, ylim_ratio=ylim_ratio, **kwargs)

    return fig, ax
