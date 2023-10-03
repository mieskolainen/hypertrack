# IO-tools
#
# m.mieskolainen@imperial.ac.uk, 2023

import re
import glob
import os
import psutil
import numpy as np
import torch
from termcolor import cprint
from typing import List
import subprocess
from datetime import datetime
import socket

def construct_event_range(event_start, event_end):
    """
    Construct event range list
    
    Args:
        event_start:  starting index (integer or list of integers)
        event_end:    ending index
    
    Returns:
        event index list
    """
    
    if type(event_start) is int:
        event_range = np.arange(event_start, event_end+1)
    else:
        event_range = np.array([], dtype=int)
        
        if len(event_start) != len(event_end):
            raise Exception(__name__ + '.construct_event_range: len(event_start) != len(event_end)')

        for i in range(len(event_start)):
            event_range = np.concatenate([event_range, np.arange(event_start[i], event_end[i]+1)])
    
    return event_range

def load_models(model: dict, epoch: int, read_tag: str, device: torch.device,
                keys: List[str]=None, mode='eval', path: str=None):
    """
    Load the latest timestamp saved models
    
    Args:
        model:     neural models in a dictionary
        epoch:     epoch number (or iteration)
        read_tag:  model tag name to be read out
        device:    torch device
        keys:      model dictionary keys
        mode:      'train' or 'eval'
        path:      path to model files
    
    Returns:
        model:       model dictionary
        found_epoch: loaded epoch savetag per model
    """
    
    if path is None:
        CWD  = os.getcwd()
        path = f'{CWD}'
    
    path = path + f'/models/tag_{read_tag}'
    
    if keys is None: keys = model.keys()
    
    found_epoch = {} # Save filenames
    
    for m in keys:
        
        if epoch == -1:
            list_of_files = glob.glob(f'{path}/model_{m}*') # * ~ take all
            if len(list_of_files) == 0:
                raise Exception(f'Did not found any model files under {path}')
            file = max(list_of_files, key=os.path.getctime)
        else:
            file = f'{path}/model_{m}_epoch_{epoch}.pt'
        
        found_epoch[m] = file
        
        print(__name__ + f'.load_models: Loading torch model [{m}]: {file} ({mode})')
        checkpoint = torch.load(file, map_location=device)
        model[m].load_state_dict(checkpoint['model'], strict=False)
        model[m].to(device)
        
        if   mode == 'eval':
            model[m].eval()
        elif mode == 'train':
            model[m].train()
        else:
            raise Exception(__name__ + '.load_models: Unknown mode (use eval or train)')
    
    return model, found_epoch

def grab_torch_file(key:str, name: str, save_tag: str, epoch: int, path: str=None):
    """
    Grab a torch file from a disk
    
    Args:
        key:         file key identifier
        name:        file name identifier
        save_tag:    model tag
        epoch:       epoch number (or iteration), set -1 for the latest by timestamp
        path:        base disk path
        
    Returns:
        filename     full filename found
        found_epoch: epoch number (or iteration)
    """
    if path is None:
        CWD  = os.getcwd()
        path = f'{CWD}'
    
    path = path + f'/models/tag_{save_tag}/'
    list_of_files = glob.glob(f'{path}/{name}_{key}*') # * ~ take all
    
    if len(list_of_files) == 0:
        raise Exception(__name__ + '.load_models: No any saved models found.')

    if epoch == -1: # Take latest by timestamp
        filename = max(list_of_files, key=os.path.getctime)
        found_epoch = int(re.search('.*_(.*).pt', filename).group(1))
    else:
        filename = f'{path}/{name}_{key}_epoch_{epoch}.pt'
        found_epoch = epoch
        
    return filename, found_epoch

def sysinfo():
    """
    Returns system info string
    """
    total = psutil.virtual_memory()[0] / 1024**3
    free  = psutil.virtual_memory()[1] / 1024**3
    return f'{datetime.now()} | hostname: {socket.gethostname()} | CPU cores: {os.cpu_count()} | RAM: {total:0.1f} ({free:0.1f}) GB'

def showmem(color: str='red'):
    """
    Print CPU memory use
    """
    cprint(__name__ + f""".showmem: Process RAM: {process_memory_use():0.2f} GB [total RAM in use {psutil.virtual_memory()[2]} %]""", color)

def showmem_cuda(device: torch.device, color: str='red'):
    """
    Print CUDA memory use
    
    Args:
        device:  torch device
        color:   print color
    """
    cprint(__name__ + f".showmem_cuda: Process RAM: {process_memory_use():0.2f} GB [total RAM in use {psutil.virtual_memory()[2]} %] | VRAM usage: {get_gpu_memory_map()} GB [total VRAM {torch_cuda_total_memory(device):0.2f} GB]", color)

def get_gpu_memory_map():
    """Get the GPU VRAM use in GB.
    
    Returns:
        dictionary with keys as device ids [integers]
        and values the memory used by the GPU.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')

    # into dictionary
    gpu_memory = [int(x)/1024.0 for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def torch_cuda_total_memory(device: torch.device):
    """
    Return CUDA device VRAM available in GB.
    
    Args:
        device: torch device
    """
    return torch.cuda.get_device_properties(device).total_memory / 1024.0**3

def process_memory_use():
    """
    Return system memory (RAM) used by the process in GB.
    """
    pid = os.getpid()
    py  = psutil.Process(pid)
    return py.memory_info()[0]/2.**30
