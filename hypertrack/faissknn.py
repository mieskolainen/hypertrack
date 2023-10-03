# Wrapper class for faiss kNN (compare with Voxel-Dynamics)
#
# m.mieskolainen@imperial.ac.uk, 2023

import faiss
import numpy as np


class faisskNN:

    def __init__(self, device='cpu'):
        self.index  = None
        self.device = device

    def fit(self, X, nlist=128, nprobe=8, IVF_on=True, metric='L2'):
        """
        Args:
            X     :  Data array (N x dim)
            nlist :  Number of cells / clusters to partition data into
            nprobe:  Number of nearest cells to include in the search (default 1)
            IVF_on:  Approximate (fast) IVF search
            metric:  Comparison type 'L2' (Euclidian distance) or 'IP' (inner product)
        """
        
        dim = X.shape[1]

        # How the vectors will be stored / compared
        if   metric == 'L2': # L2-metric
            self.index = faiss.IndexFlatL2(dim)
        elif metric == 'IP': # Inner product
            self.index = faiss.IndexFlatIP(dim)
        else:
            raise Exception(__name__ + f'fit: Unknown "metric" {metric}')
        
        # Approximate search IVF
        if IVF_on:
            self.index = faiss.IndexIVFFlat(self.index, dim, nlist)

        # GPU version
        if 'cuda' in self.device:
            self.res       = faiss.StandardGpuResources()  # use a single GPU
            self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
            self.gpu_index.train(X.astype(np.float32))
            self.gpu_index.add(X.astype(np.float32))

        # CPU version
        else:
            self.index.train(X.astype(np.float32))
            self.index.add(X.astype(np.float32))

        self.index.nprobe = nprobe
        #print(f'nlist = {nlist}, nprobe = {self.index.nprobe}')


    def search(self, xq, k):
        """
        Args:
            xq:  query vectors array (N x dim)
            k :  number of nearest neigbours to search
        Returns:
            D,I: distances, indices
        """
        if 'cuda' in self.device:
            D, I = self.gpu_index.search(xq.astype(np.float32), k)
            D = D.astype(float)
            I = I.astype(int)
        else:
            D, I = self.index.search(xq.astype(np.float32), k)
        
        return D,I
