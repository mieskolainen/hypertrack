# Simple transforms
#
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np


def transform_r_eta_phi(vec):
    """
    Transform (x,y,z) -> (r,eta,phi)

    Args:
        vec:  cartesian coordinate array (N x 3)
    
    Returns:
        coordinate array
    """
    x,y,z    = vec[:,0],vec[:,1],vec[:,2]
    
    r        = np.sqrt(x**2 + y**2)
    theta    = np.arctan2(r, z)
    cosTheta = np.cos(theta)
    
    # r, eta, phi
    out      = np.zeros_like(vec)
    
    out[:,0] = r
    out[:,1] = -0.5*np.log((1.0 - cosTheta) / (1.0 + cosTheta))
    out[:,2] = np.arctan2(y,x)
    
    return out
