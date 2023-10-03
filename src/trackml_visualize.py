# Visualize hits of tracks of TrackML dataset
#
# m.mieskolainen@imperial.ac.uk, 2023

import sys
sys.path.append('../hypertrack')

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hypertrack import trackml, visualize, transforms

CWD = os.getcwd()

OUT_PATH = f'{CWD}/figs/visualize/'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

event = 1000

X, x_ind, objects, hit_weights = trackml.load_reduce_event(event=event)

ind = np.arange(20) # Track indices

print(f'Event: {event}')

for i in ind:
    
    pvec = transforms.transform_r_eta_phi(objects[i,3:6][None,:]).squeeze()
    
    # Print
    print(f'Track:    {i}')
    print(f'')
    print(f'Vertex:   {objects[i,0:3]}')
    print(f'Momentum: {objects[i,3:6]} | (pt,eta,phi) = {pvec}')
    print(f'Charge:   {int(objects[i,6])}')
    print(f'Hits:     {int(objects[i,7])} ({np.sum(x_ind[i,:] != -1)} in x_ind[{i},:])')
    print('')
    print(x_ind[i,:])
    
    # Visualize (hits are not in causal time-order necessarily)
    title = f'event: {event}, track: {i} ($p_T = {pvec[0]:0.1f}, \\eta = {pvec[1]:0.1f}, \\phi = {pvec[2]:0.1f}$)'
    
    fig,ax = visualize.plot_tracks(X=X, x_ind=x_ind, ind=[i], title=title)
    plt.savefig(f'figs/visualize/event_{event}_track_{i}.pdf', bbox_inches='tight')
    plt.close()
    
    nhits = np.sum(x_ind[i,:] != -1)
    for k in range(nhits):
        index = x_ind[i,k]
        print(f'hit[{k:2d}] coord: ({X[index,0]:4.3f}, {X[index,1]:4.3f}, {X[index,2]:4.3f})')

    print('')
    print('----------------------------------------------------')

