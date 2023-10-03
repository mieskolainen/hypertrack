# HyperTrack neural model torch classes
#
# m.mieskolainen@imperial.ac.uk, 2023

from typing import Callable, Union, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T

from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Size, Tensor, OptTensor, PairTensor, PairOptTensor, OptPairTensor, Adj

from hypertrack.dmlp import MLP
import hypertrack.flows as fnn


class SuperEdgeConv(MessagePassing):
    r"""
    Custom GNN convolution operator aka 'generalized EdgeConv' (original EdgeConv: arxiv.org/abs/1801.07829)
    """
    
    def __init__(self, mlp_edge: Callable, mlp_latent: Callable, aggr: str='mean',
                 mp_attn_dim: int=0, use_residual=True, **kwargs):
        
        if aggr == 'multi-aggregation':
            aggr = torch_geometric.nn.aggr.MultiAggregation(aggrs=['sum', 'mean', 'std', 'max', 'min'], mode='attn',
                    mode_kwargs={'in_channels': mp_attn_dim, 'out_channels': mp_attn_dim, 'num_heads': 1})
        
        if aggr == 'set-transformer':
            aggr = torch_geometric.nn.aggr.SetTransformerAggregation(channels=mp_attn_dim, num_seed_points=1, 
                    num_encoder_blocks=1, num_decoder_blocks=1, heads=1, concat=False,
                    layer_norm=False, dropout=0.0)

        super().__init__(aggr=aggr, **kwargs)
        self.nn           = mlp_edge
        self.nn_final     = mlp_latent
        self.use_residual = use_residual
        
        self.reset_parameters()

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            #print(__name__ + f'.SuperEdgeConv: Initializing module: {module}')
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    def reset_parameters(self):
        torch_geometric.nn.inits.reset(self.nn)
        torch_geometric.nn.inits.reset(self.nn_final)
    
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
            edge_attr: OptTensor = None, edge_weight: OptTensor = None, size: Size = None) -> Tensor:

        if edge_attr is not None and len(edge_attr.shape) == 1: # if 1-dim edge_attributes
            edge_attr = edge_attr[:,None]
        
        # Message passing
        m = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight, size=None)
        
        # Final MLP
        y = self.nn_final(torch.concat([x, m], dim=-1))
        
        # Residual connections
        if self.use_residual and (y.shape[-1] == x.shape[-1]):
            y = y + x
        
        return y

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor, edge_weight: OptTensor) -> Tensor:
        
        # Edge features
        e1 = torch.norm(x_j - x_i, dim=-1) # Norm of the difference (invariant under rotations and translations)
        e2 = torch.sum(x_j * x_i,  dim=-1) # Dot-product (invariant under rotations but not translations)
        
        if len(e1.shape) == 1:
            e1 = e1[:,None]
            e2 = e2[:,None]
        
        if edge_attr is not None:
            m = self.nn(torch.cat([x_i, x_j - x_i, x_j * x_i, e1, e2, edge_attr], dim=-1))
        else:
            m = self.nn(torch.cat([x_i, x_j - x_i, x_j * x_i, e1, e2], dim=-1))
        
        return m if edge_weight is None else m * edge_weight.view(-1, 1)
    
    def __repr__(self):
        return f'{self.__class__.__name__} (nn={self.nn}, nn_final={self.nn_final})'


class CoordNorm(nn.Module):
    """
    Coordinate normalization for stability with GaugeEdgeConv
    """
    def __init__(self, eps = 1e-8, scale_init = 1.0):
        super().__init__()
        self.eps   = eps
        scale      = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coord):
        norm = coord.norm(dim = -1, keepdim = True)
        normed_coord = coord / norm.clamp(min = self.eps)
        return normed_coord * self.scale


class GaugeEdgeConv(MessagePassing):
    r"""
    Custom GNN convolution operator aka 'E(N) equivariant GNN' (arxiv.org/abs/2102.09844)
    """
    
    def __init__(self, mlp_edge: Callable, mlp_coord: Callable, mlp_latent: Callable, coord_dim: int=0,
        update_coord: bool=True, update_latent: bool=True, aggr: str='mean', mp_attn_dim: int=0, 
        norm_coord=False, norm_coord_scale_init = 1e-2, **kwargs):
        
        kwargs.setdefault('aggr', aggr)
        super(GaugeEdgeConv, self).__init__(**kwargs)
        
        self.mlp_edge      = mlp_edge
        self.mlp_coord     = mlp_coord
        self.mlp_latent    = mlp_latent
        
        self.update_coord  = update_coord
        self.update_latent = update_latent
        
        self.coord_dim     = coord_dim
        
        # Coordinate normalization
        self.coors_norm = CoordNorm(scale_init = norm_coord_scale_init) if norm_coord else nn.Identity()
        
        self.reset_parameters()

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            #print(__name__ + f'.GaugeEdgeConv: Initializing module: {module}')
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    def reset_parameters(self):
        torch_geometric.nn.inits.reset(self.mlp_edge)
        torch_geometric.nn.inits.reset(self.mlp_coord)
        torch_geometric.nn.inits.reset(self.mlp_latent)
        

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
            edge_attr: OptTensor = None, edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """
        Forward function
        """
        
        # Separate spatial (e.g. 3D-coordinates) and features
        coord, feats = x[..., 0:self.coord_dim], x[..., self.coord_dim:]
        
        # Coordinate difference: x_i - x_j
        diff_coord = coord[edge_index[0]] - coord[edge_index[1]]
        diff_norm2 = (diff_coord ** 2).sum(dim=-1, keepdim=True)
        
        if edge_attr is not None:
            if len(edge_attr.shape) == 1: # if 1-dim edge_attributes
                edge_attr = edge_attr[:,None]

            edge_attr_feats = torch.cat([edge_attr, diff_norm2], dim=-1)
        else:
            edge_attr_feats = diff_norm2
        
        # Propagation
        latent_out, coord_out = self.propagate(edge_index, x=feats, edge_attr=edge_attr_feats,
                                    edge_weight=edge_weight, coord=coord, diff_coord=diff_coord, size=None)

        return torch.cat([coord_out, latent_out], dim=-1)
    
    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor, edge_weight: OptTensor) -> Tensor:
        """
        Message passing core operation between nodes (i,j)
        """
        
        m_ij = self.mlp_edge(torch.cat([x_i, x_j, edge_attr], dim=-1))

        return m_ij if edge_weight is None else m_ij * edge_weight.view(-1, 1)

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """
        The initial call to start propagating messages.
        
        Args:
            edge_index: holds the indices of a general (sparse)
                        assignment matrix of shape :obj:`[N, M]`.
            size:       (tuple, optional) if none, the size will be inferred
                        and assumed to be quadratic.
            **kwargs:   Any additional data which is needed to construct and
                        aggregate messages, and to update node embeddings.
        """
        
        # Check:
        # https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/message_passing.py
        
        size          = self._check_input(edge_index, size)
        coll_dict     = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs    = self.inspector.distribute('message',   coll_dict)
        aggr_kwargs   = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update',    coll_dict)
        
        # Message passing of node latent embeddings
        m_ij = self.message(**msg_kwargs)
        
        if self.update_coord:
            
            # Normalize
            kwargs["diff_coord"] = self.coors_norm(kwargs["diff_coord"])
            
            # Aggregate weighted coordinates
            mhat_i    = self.aggregate(kwargs["diff_coord"] * self.mlp_coord(m_ij), **aggr_kwargs)
            coord_out = kwargs["coord"] + mhat_i # Residual connection
            
        else:
            coord_out = kwargs["coord"]
        
        
        if self.update_latent:
            
            # Aggregate message passing results
            m_i        = self.aggregate(m_ij, **aggr_kwargs)
            
            # Update latent representation
            latent_out = self.mlp_latent(torch.cat([kwargs["x"], m_i], dim = -1))
            latent_out = kwargs["x"] + latent_out # Residual connection
        else:
            latent_out = kwargs["x"]
        
        # Return tuple
        return self.update((latent_out, coord_out), **update_kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__}(GaugeEdgeConv = {self.mlp_edge} | {self.mlp_coord} | {self.mlp_latent})'


class InverterNet(torch.nn.Module):
    """
    HyperTrack neural model "umbrella" class, encapsulating GNN and Transformer etc.
    """
    def __init__(self, graph_block_param={}, cluster_block_param={}):
        
        """
        conv_aggr: 'mean' seems very crucial.
        """
        super().__init__()
        
        self.training_on   = True
        
        self.coord_dim     = graph_block_param['coord_dim']   # Input dimension
        self.h_dim         = graph_block_param['h_dim']       # Intermediate latent dimension
        self.z_dim         = graph_block_param['z_dim']       # Final latent dimension
        
        self.GNN_model     = graph_block_param['GNN_model']
        self.nstack        = graph_block_param['nstack']
        
        self.edge_type     = graph_block_param['edge_type']
        MLP_fusion         = graph_block_param['MLP_fusion']
        
        self.num_edge_attr = 1 # cf. custom edge features constructed in self.encode()
        self.conv_gnn_edx  = nn.ModuleList()
        
        # Transformer node mask learnable "soft" threshold
        self.thr = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.thr, 0.5)
        
        # 1. GNN encoder
        
        ## Model type
        if self.GNN_model == 'GaugeEdgeConv':
            
            self.m_dim     = graph_block_param['GaugeEdgeConv']['m_dim']

            MLP_GNN_edge   = graph_block_param['MLP_GNN_edge']
            MLP_GNN_coord  = graph_block_param['MLP_GNN_coord']
            MLP_GNN_latent = graph_block_param['MLP_GNN_latent']
            
            num_intrinsic_attr = 1 # distance
            
            for i in range(self.nstack):
                self.conv_gnn_edx.append(GaugeEdgeConv(
                    mlp_edge    = MLP([2*self.h_dim + self.num_edge_attr + num_intrinsic_attr, 2*self.m_dim, self.m_dim], **MLP_GNN_edge),
                    mlp_coord   = MLP([self.m_dim, 2*self.m_dim, 1], **MLP_GNN_coord),
                    mlp_latent  = MLP([self.h_dim + self.m_dim, 2*self.h_dim, self.h_dim], **MLP_GNN_latent),
                    aggr=graph_block_param['GaugeEdgeConv']['aggr'][i],
                    norm_coord=graph_block_param['GaugeEdgeConv']['norm_coord'],
                    norm_coord_scale_init=graph_block_param['GaugeEdgeConv']['norm_coord_scale_init'],
                    coord_dim=self.coord_dim))

            self.mlp_fusion_edx = MLP([self.nstack * (self.coord_dim + self.h_dim), self.h_dim, self.z_dim], **MLP_fusion)
        
        ## Model type
        elif self.GNN_model == 'SuperEdgeConv':
            
            self.m_dim     = graph_block_param['SuperEdgeConv']['m_dim']
            
            MLP_GNN_edge   = graph_block_param['MLP_GNN_edge']
            MLP_GNN_latent = graph_block_param['MLP_GNN_latent']
            
            num_intrinsic_attr = 2 # distance and dot-product
            
            self.conv_gnn_edx.append(SuperEdgeConv(
                    mlp_edge    = MLP([3*self.coord_dim + self.num_edge_attr + num_intrinsic_attr, self.m_dim, self.m_dim], **MLP_GNN_edge),
                    mlp_latent  = MLP([self.coord_dim + self.m_dim, self.h_dim, self.h_dim], **MLP_GNN_latent),
                    mp_attn_dim=self.h_dim, aggr=graph_block_param['SuperEdgeConv']['aggr'][0], use_residual=False))
            
            for i in range(1,self.nstack):
                self.conv_gnn_edx.append(SuperEdgeConv(
                    mlp_edge    = MLP([3*self.h_dim + self.num_edge_attr + num_intrinsic_attr, self.m_dim, self.m_dim], **MLP_GNN_edge),
                    mlp_latent  = MLP([self.h_dim + self.m_dim, self.h_dim, self.h_dim], **MLP_GNN_latent),
                    mp_attn_dim=self.h_dim, aggr=graph_block_param['SuperEdgeConv']['aggr'][i], use_residual=graph_block_param['SuperEdgeConv']['use_residual']))
            
            self.mlp_fusion_edx = MLP([self.nstack * self.h_dim, self.h_dim, self.z_dim], **MLP_fusion)

        else:
            raise Exception(__name__ + '.__init__: Unknown GNN_model chosen')
        
        # 2. Graph edge predictor
        MLP_correlate = graph_block_param['MLP_correlate']
        
        if   self.edge_type == 'symmetric-dot':
            self.mlp_2pt_edx = MLP([self.z_dim,   self.z_dim//2, self.z_dim//2, 1], **MLP_correlate)
        elif self.edge_type == 'symmetrized' or self.edge_type == 'asymmetric':
            self.mlp_2pt_edx = MLP([2*self.z_dim, self.z_dim//2, self.z_dim//2, 1], **MLP_correlate)
        
        # 3. Clustering predictor
        self.transformer_ccx = STransformer(**cluster_block_param)
    
    def encode(self, x, edge_index):
        """
        Encoder GNN
        """
        # Compute node degree 'custom feature' between edges
        d         = torch_geometric.utils.degree(edge_index[0,:])
        edge_attr = (d[edge_index[0,:]] - d[edge_index[1,:]]) / torch.mean(d)
        edge_attr = edge_attr.to(x.dtype)
        
        # We take each output for the parallel fusion
        x_out = [None] * self.nstack
        
        # First input [x; one-vector for the first embeddings (latents)]
        if self.GNN_model == 'GaugeEdgeConv':
            x_ = torch.cat([x, torch.ones((x.shape[0], self.h_dim), device=x.device)], dim=-1)
        else:
            x_ = x
        
        # Apply GNN layers
        x_out[0] = self.conv_gnn_edx[0](x_, edge_index, edge_attr)
        for i in range(1,self.nstack):
            x_out[i] = self.conv_gnn_edx[i](x_out[i-1], edge_index, edge_attr)
        
        return self.mlp_fusion_edx(torch.cat(x_out, dim=-1))
    
    def decode(self, z, edge_index):
        """
        Decoder of two-point correlations (edges)
        """
        if   self.edge_type == 'symmetric-dot':
            return self.mlp_2pt_edx(z[edge_index[0],:] * z[edge_index[1],:])
        
        elif self.edge_type == 'symmetrized':
            a = self.mlp_2pt_edx(torch.cat([z[edge_index[0],:], z[edge_index[1],:]], dim=-1))
            b = self.mlp_2pt_edx(torch.cat([z[edge_index[1],:], z[edge_index[0],:]], dim=-1))
            return (a + b) / 2.0

        elif self.edge_type == 'asymmetric':
            a = self.mlp_2pt_edx(torch.cat([z[edge_index[0],:], z[edge_index[1],:]], dim=-1))
            return a

    def decode_cc_ind(self, X, X_pivot, X_mask=None, X_pivot_mask=None):
        """
        Decoder of N-point node mask
        """
        
        return self.transformer_ccx(X=X, X_pivot=X_pivot, X_mask=X_mask, X_pivot_mask=X_pivot_mask)
    
    def set_model_param_grad(self, string_id='edx', requires_grad=True):
        """
        Freeze or unfreeze model parameters (for the gradient descent)
        """
        
        for name, W in self.named_parameters():   
            if string_id in name:
                W.requires_grad = requires_grad
                #print(f'Setting requires_grad={W.requires_grad} of the parameter <{name}>')
        return

    def get_model_param_grad(self, string_id='edx'):
        """
        Get model parameter state (for the gradient descent)
        """
        
        for name, W in self.named_parameters():   
            if string_id in name: # Return the state of the first
                return W.requires_grad


class MAB(nn.Module):
    """
    Attention based set Transformer block (arxiv.org/abs/1810.00825)
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads=4, ln=True, dropout=0.0,
                 MLP_param={'act': 'relu', 'bn': False, 'dropout': 0.0, 'last_act': True}):
        super(MAB, self).__init__()
        assert dim_V % num_heads == 0, "MAB: dim_V must be divisible by num_heads"
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.W_q = nn.Linear(dim_Q, dim_V)
        self.W_k = nn.Linear(dim_K, dim_V)
        self.W_v = nn.Linear(dim_K, dim_V)
        self.W_o = nn.Linear(dim_V, dim_V) # Projection layer
        
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        
        if dropout > 0:
            self.Dropout = nn.Dropout(dropout)
        
        self.MLP = MLP([dim_V, dim_V, dim_V], **MLP_param)
    
        # We use torch default initialization here
    
    def reshape_attention_mask(self, Q, K, mask):
        """
        Reshape attention masks
        """
        total_mask = None

        if mask[0] is not None:
            qmask = mask[0].repeat(self.num_heads,1)[:,:,None]  # New shape = [# heads x # batches, # queries, 1]

        if mask[1] is not None:
            kmask = mask[1].repeat(self.num_heads,1)[:,None,:]  # New shape = [# heads x # batches, 1, # keys]
        
        if   mask[0] is None and mask[1] is not None:
            total_mask = kmask.repeat(1,Q.shape[1],1)
        elif mask[0] is not None and mask[1] is None:
            total_mask = qmask.repeat(1,1,K.shape[1])
        elif mask[0] is not None and mask[1] is not None:
            total_mask = qmask & kmask # will auto broadcast dimensions to [# heads x # batches, # queries, # keys]
        
        return total_mask

    def forward(self, Q, K, mask = (None, None)):
        """
        Q:    queries
        K:    keys
        mask: query mask [#batches x #queries], keys mask [#batches x #keys]
        """
        dim_split = self.dim_V // self.num_heads
        
        # Apply Matrix-vector multiplications
        # and do multihead splittings (reshaping)
        Q_ = torch.cat(self.W_q(Q).split(dim_split, -1), 0)
        K_ = torch.cat(self.W_k(K).split(dim_split, -1), 0)
        V_ = torch.cat(self.W_v(K).split(dim_split, -1), 0)
        
        # Dot-product attention: softmax(QK^T / sqrt(dim_V))
        QK = Q_.bmm(K_.transpose(-1,-2)) / math.sqrt(self.dim_V)  # bmm does batched matrix multiplication
        
        # Attention mask
        total_mask = self.reshape_attention_mask(Q=Q, K=K, mask=mask)
        if total_mask is not None:                 
            QK.masked_fill_(~total_mask, float('-1E6'))
        
        # Compute attention probabilities
        A = torch.softmax(QK,-1)
        
        # Residual connection of Q + multi-head attention A weighted V result
        H = Q + self.W_o(torch.cat((A.bmm(V_)).split(Q.size(0), 0),-1))
        
        # First layer normalization + Dropout
        H = H if getattr(self, 'ln0', None) is None else self.ln0(H)
        H = H if getattr(self, 'Dropout', None) is None else self.Dropout(H)
        
        # Residual connection of H + feed-forward net
        H = H + self.MLP(H)
        
        # Second layer normalization + Dropout
        H = H if getattr(self, 'ln1', None) is None else self.ln1(H)
        H = H if getattr(self, 'Dropout', None) is None else self.Dropout(H)
        
        return H


class SAB(nn.Module):
    """
    Full self-attention MAB(X,X)
    ~O(N^2)
    """
    def __init__(self, dim_in, dim_out, num_heads=4, ln=True, dropout=0.0,
                 MLP_param={'act': 'relu', 'bn': False, 'dropout': 0.0, 'last_act': True}):
        super(SAB, self).__init__()
        self.mab = MAB(dim_Q=dim_in, dim_K=dim_in, dim_V=dim_out, 
                       num_heads=num_heads, ln=ln, dropout=dropout, MLP_param=MLP_param)

    def forward(self, X, mask=None):
        return self.mab(Q=X, K=X, mask=(mask, mask))

class ISAB(nn.Module):
    """
    Faster version of SAB with inducing points
    """
    def __init__(self, dim_in, dim_out, num_inds, num_heads=4, ln=True, dropout=0.0,
                 MLP_param={'act': 'relu', 'bn': False, 'dropout': 0.0, 'last_act': True}):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_Q=dim_out, dim_K=dim_in, dim_V=dim_out,
                        num_heads=num_heads, ln=ln, dropout=dropout, MLP_param=MLP_param)
        self.mab1 = MAB(dim_Q=dim_in, dim_K=dim_out, dim_V=dim_out,
                        num_heads=num_heads, ln=ln, dropout=dropout, MLP_param=MLP_param)

    def forward(self, X, mask=None):
        H = self.mab0(Q=self.I.expand(X.size(0),-1,-1), K=X, mask=(None, mask))
        H = self.mab1(Q=X, K=H, mask=(mask, None))
        return H

class PMA(nn.Module):
    """
    Adaptive pooling with "k" > 1 option (several learnable reference vectors)
    """
    def __init__(self, dim, k=1, num_heads=4, ln=True, dropout=0.0,
                 MLP_param={'act': 'relu', 'bn': False, 'dropout': 0.0, 'last_act': True}):
        super(PMA, self).__init__()
        self.S   = nn.Parameter(torch.Tensor(1, k, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim_Q=dim, dim_K=dim, dim_V=dim,
                       num_heads=num_heads, ln=ln, dropout=dropout, MLP_param=MLP_param)
    
    def forward(self, X, mask):
        return self.mab(Q=self.S.expand(X.size(0),-1,-1), K=X, mask=(None, mask))

class STransformer(nn.Module):
    """
    Set Transformer based clustering network
    """

    class mySequential(nn.Sequential):
        """
        Multiple inputs version of nn.Sequential customized for
        multiple self-attention layers with a (same) mask
        """
        def forward(self, *inputs):
            
            X, mask = inputs[0], inputs[1]
            for module in self._modules.values():
                X = module(X,mask)
            return X
    
    def __init__(self, in_dim, h_dim, output_dim, nstack_dec=4, 
                 MLP_enc={}, MAB_dec={}, SAB_dec={}, MLP_mask={}):
        
        super().__init__()
        
        # Encoder MLP
        self.MLP_E = MLP([in_dim, in_dim, h_dim], **MLP_enc)
        
        # Decoder
        self.mab_D = MAB(dim_Q=h_dim, dim_K=h_dim, dim_V=h_dim, **MAB_dec)

        # Decoder self-attention layers
        self.sab_stack_D = self.mySequential(*[
            self.mySequential(
                SAB(dim_in=h_dim, dim_out=h_dim, **SAB_dec)
            )
            for i in range(nstack_dec)
        ])
        
        # Final mask MLP
        self.MLP_m = MLP([h_dim, h_dim//2, h_dim//4, output_dim], **MLP_mask)
    
    def forward(self, X, X_pivot, X_mask = None, X_pivot_mask = None):
        """
        X:            input data vectors per row
        X_pivot:      pivotal data (at least one per batch)
        X_mask:       boolean (batch) mask for X (set 0 for zero-padded null elements)
        X_pivot_mask: boolean (batch) mask for X_pivot
        """
        
        # Simple encoder
        G       = self.MLP_E(X)
        G_pivot = self.MLP_E(X_pivot)
        
        # Compute cross-attention and self-attention
        H_m = self.sab_stack_D(self.mab_D(Q=G, K=G_pivot, mask=(X_mask, X_pivot_mask)), X_mask)
        
        # Decode logits
        return self.MLP_m(H_m)

class TrackFlowNet(torch.nn.Module):
    """
    Normalizing Flow Network [experimental]
    """
    def __init__(self, in_dim, num_cond_inputs=None, h_dim=64, nblocks=4, act='tanh'):
        """
        conv_aggr: 'mean' in GNN seems to work ok with Flow!
        """
        super().__init__()

        self.training_on = True
        
        # -----------------------------------
        # MAF density estimator

        modules = []
        for _ in range(nblocks):
            modules += [
                fnn.MADE(num_inputs=in_dim, num_hidden=h_dim, num_cond_inputs=num_cond_inputs, act=act),
                #fnn.BatchNormFlow(in_dim), # May cause problems in recursive use
                fnn.Reverse(in_dim)
            ]
        
        self.track_pdf = fnn.FlowSequential(*modules)
        # -----------------------------------

    def set_model_param_grad(self, string_id='pdf', requires_grad=True):
        """
        Freeze or unfreeze model parameters (for the gradient descent)
        """
        
        for name, W in self.named_parameters():   
            if string_id in name:
                W.requires_grad = requires_grad
                #print(f'Setting requires_grad={W.requires_grad} of the parameter <{name}>')
        return
