import numpy as np
import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch

from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE


class GPSLayerMPNN(nn.Module):
    """Local MPNN + full graph attention x-former layer."""

    def __init__(
        self,
        dim_h,
        local_gnn_type,
        global_model_type,
        num_heads,
        act="relu",
        pna_degrees=None,
        equivstable_pe=False,
        dropout=0.0,
        attn_dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        bigbird_cfg=None,
        log_attn_weights=False,
    ):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in [
            "Transformer",
            "BiasedTransformer",
        ]:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )

        # Local message-passing model.
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == "None":
            self.local_model = None

        # MPNNs without edge attributes support.
        elif local_gnn_type == "GCN":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == "GIN":
            self.local_gnn_with_edge_attr = False
            gin_nn = nn.Sequential(
                Linear_pyg(dim_h, dim_h), self.activation(), Linear_pyg(dim_h, dim_h)
            )
            self.local_model = pygnn.GINConv(gin_nn)

        # MPNNs supporting also edge attributes.
        elif local_gnn_type == "GENConv":
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == "GINE":
            gin_nn = nn.Sequential(
                Linear_pyg(dim_h, dim_h), self.activation(), Linear_pyg(dim_h, dim_h)
            )
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == "GAT":
            self.local_model = pygnn.GATConv(
                in_channels=dim_h,
                out_channels=dim_h // num_heads,
                heads=num_heads,
                edge_dim=dim_h,
            )
        elif local_gnn_type == "PNA":
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ["mean", "max", "sum"]
            scalers = ["identity"]
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(
                dim_h,
                dim_h,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=min(128, dim_h),
                towers=1,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        elif local_gnn_type == "CustomGatedGCN":
            self.local_model = GatedGCNLayer(
                dim_h,
                dim_h,
                dropout=dropout,
                residual=True,
                act=act,
                equivstable_pe=equivstable_pe,
            )
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Learnable weights for combining outputs from local and global models.
        self.h_out_weights = nn.Parameter(torch.ones(2))

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
            # self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == "CustomGatedGCN":
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(
                    Batch(
                        batch=batch,
                        x=h,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        pe_EquivStableLapPE=es_data,
                    )
                )
                # GatedGCN does residual connection and dropout internally.
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.local_gnn_with_edge_attr:
                    if self.equivstable_pe:
                        h_local = self.local_model(
                            h,
                            batch.edge_index,
                            batch.edge_attr,
                            batch.pe_EquivStableLapPE,
                        )
                    else:
                        h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
                else:
                    h_local = self.local_model(h, batch.edge_index)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)


        h = h_out_list[0]

        # print("-----------")
        # print(h_out_weights_softmax[0])
        # print(h_out_weights_softmax[1])
        # print("-----------")

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        if not self.log_attn_weights:
            x = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block."""
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = (
            f"summary: dim_h={self.dim_h}, "
            f"heads={self.num_heads}"
        )
        return s
