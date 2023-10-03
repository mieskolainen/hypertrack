# Deep multilayer perceptron (MLP / FF) model wrappers
#
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def get_act(act: str):
    """
    Returns torch activation function
    
    Args:
        act:  activation function 'relu', 'tanh', 'silu', 'elu
    """
    if   act == 'relu':
        return nn.ReLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'silu':
        return nn.SiLU()
    elif act == 'elu':
        return nn.ELU()
    else:
        raise Exception(f'Uknown act "{act}" chosen')


def MLP(layers: List[int], act: str='relu', bn: bool=False, dropout: float=0.0, last_act: bool=False):
    """
    Return a Multi Layer Perceptron with an arbitrary number of layers.
    
    Args:
        layers     : input structure, such as [128, 64, 64] for a 3-layer network.
        act        : activation function
        bn         : batch normalization
        dropout    : dropout regularization
        last_act   : apply activation function after the last layer
    
    Returns:
        nn.sequential object
    """
    print(__name__ + f'.MLP: Using {act} act')

    if not last_act: # Without activation after the last layer
        
        if bn:
            return nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(layers[i - 1], layers[i]),
                    get_act(act),
                    nn.BatchNorm1d(layers[i]),
                    nn.Dropout(dropout, inplace=False)
                )
                for i in range(1,len(layers) - 1)
            ],
                nn.Linear(layers[-2], layers[-1]) # N.B. Last without act!
            )
        else:
            return nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(layers[i - 1], layers[i]),
                    get_act(act),
                    nn.Dropout(dropout, inplace=False)
                )
                for i in range(1,len(layers) - 1)
            ], 
                nn.Linear(layers[-2], layers[-1]) # N.B. Last without act!
            )
    else:
      
        if bn:
            return nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(layers[i - 1], layers[i]),
                    get_act(act),
                    nn.BatchNorm1d(layers[i]),
                    nn.Dropout(dropout, inplace=False),
                )
                for i in range(1,len(layers))
            ]
            )
        
        else:
            return nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(layers[i - 1], layers[i]),
                    get_act(act),
                    nn.Dropout(dropout, inplace=False)
                )
                for i in range(1,len(layers))
            ]
            )  

