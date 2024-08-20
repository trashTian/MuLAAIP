# -*- coding: UTF-8 -*-

from torch_geometric.nn import inits, MessagePassing, SAGPooling
from torch_geometric.nn import radius_graph
from features import d_angle_emb, d_theta_phi_emb
from torch_scatter import scatter
from torch_sparse import matmul
from torch_geometric.nn import TransformerConv, GATConv, GATv2Conv
import torch
from torch import nn
from torch.nn import Embedding
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F

import numpy as np

num_aa_type = 20
num_side_chain_embs = 8
num_bb_embs = 6


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(F.softplus(x))


class Linear(torch.nn.Module):

    def __init__(self, in_channels, out_channels, bias=True, weight_initializer='glorot'):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_initializer == 'glorot':
            inits.glorot(self.weight)
        elif self.weight_initializer == 'zeros':
            inits.zeros(self.weight)
        if self.bias is not None:
            inits.zeros(self.bias)

    def forward(self, x):
        """"""
        return F.linear(x, self.weight, self.bias)


class TwoLinear(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act=False
    ):
        super(TwoLinear, self).__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x


class InteractionBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_channels,
            output_channels,
            num_radial,
            num_spherical,
            num_layers,
            mid_emb,
            act=swish,
            # act=mish,
            num_pos_emb=16,
            dropout=0,
            level='allatom'
    ):
        super(InteractionBlock, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)

        self.conv0 = GATConv(hidden_channels, hidden_channels)
        self.conv1 = GATConv(hidden_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)

        self.lin_feature0 = TwoLinear(num_radial * num_spherical ** 2, mid_emb, hidden_channels)
        if level == 'aminoacid':
            self.lin_feature1 = TwoLinear(num_radial * num_spherical, mid_emb, hidden_channels)
        elif level == 'backbone' or level == 'allatom':
            self.lin_feature1 = TwoLinear(3 * num_radial * num_spherical, mid_emb, hidden_channels)
        self.lin_feature2 = TwoLinear(num_pos_emb, mid_emb, hidden_channels)

        self.lin_1 = Linear(hidden_channels, hidden_channels)
        self.lin_2 = Linear(hidden_channels, hidden_channels)

        self.lin0 = Linear(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.lins_cat = torch.nn.ModuleList()
        self.lins_cat.append(Linear(3 * hidden_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.lins_cat.append(Linear(hidden_channels, hidden_channels))

        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.bn0 = nn.BatchNorm1d(24)

        # all-atom
        self.bn1 = nn.BatchNorm1d(36)
        # amino
        # self.bn1 = nn.BatchNorm1d(12)

        self.bn_pos_emb = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv0.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.lin_feature0.reset_parameters()
        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        for lin in self.lins:
            lin.reset_parameters()
        for lin in self.lins_cat:
            lin.reset_parameters()

        self.final.reset_parameters()

    def forward(self, x, feature0, feature1, pos_emb, edge_index, batch):
        x = self.bn(x)
        feature0 = self.bn0(feature0)
        feature1 = self.bn1(feature1)

        x_lin_1 = self.act(self.lin_1(x))
        x_lin_2 = self.act(self.lin_2(x))

        feature0 = self.lin_feature0(feature0)
        h0 = self.conv0(x_lin_1, edge_index, feature0)
        h0 = self.lin0(h0)
        h0 = self.act(h0)
        h0 = self.bn2(h0)
        h0 = self.dropout(h0)

        feature1 = self.lin_feature1(feature1)
        h1 = self.conv1(x_lin_1, edge_index, feature1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)
        h1 = self.bn3(h1)
        h1 = self.dropout(h1)

        feature2 = self.lin_feature2(pos_emb)
        h2 = self.conv2(x_lin_1, edge_index, feature2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)
        h2 = self.bn4(h2)
        h2 = self.dropout(h2)

        h = torch.cat((h0, h1, h2), 1)
        for lin in self.lins_cat:
            h = self.act(lin(h))

        h = h + x_lin_2

        for lin in self.lins:
            h = self.act(lin(h))
        h = self.final(h)
        return h


class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x

        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


def get_degree_mat(adj_mat, pow=1, degree_version='v1'):
    degree_mat = torch.eye(adj_mat.size()[0]).to(adj_mat.device)

    if degree_version == 'v1':
        degree_list = torch.sum((adj_mat > 0), dim=1).float()
    elif degree_version == 'v2':
        adj_mat_hat = F.relu(adj_mat)
        degree_list = torch.sum(adj_mat_hat, dim=1).float()
    elif degree_version == 'v3':
        degree_list = torch.sum(adj_mat, dim=1).float()
        degree_list = F.relu(degree_list)
    else:
        exit('error degree_version ' + degree_version)
    degree_list = torch.pow(degree_list, pow)
    degree_mat = degree_mat * degree_list
    return degree_mat


def get_laplace_mat(adj_mat, type='sym', add_i=False, degree_version='v2'):
    if type == 'sym':
        # Symmetric normalized Laplacian
        if add_i is True:
            adj_mat_hat = torch.eye(adj_mat.size()[0]).to(adj_mat.device) + adj_mat
        else:
            adj_mat_hat = adj_mat
        # adj_mat_hat = adj_mat_hat[adj_mat_hat > 0]
        degree_mat_hat = get_degree_mat(adj_mat_hat, pow=-0.5, degree_version=degree_version)
        # print(degree_mat_hat.dtype, adj_mat_hat.dtype)
        laplace_mat = torch.mm(degree_mat_hat, adj_mat_hat)
        # print(laplace_mat)
        laplace_mat = torch.mm(laplace_mat, degree_mat_hat)
        return laplace_mat
    elif type == 'rw':
        # Random walk normalized Laplacian
        adj_mat_hat = torch.eye(adj_mat.size()[0]).to(adj_mat.device) + adj_mat
        degree_mat_hat = get_degree_mat(adj_mat_hat, pow=-1)
        laplace_mat = torch.mm(degree_mat_hat, adj_mat_hat)
        return laplace_mat


class GCNConvL(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 dropout=0.6,
                 bias=True
                 ):
        super(GCNConvL, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.bias = bias
        self.weight = Parameter(
            torch.Tensor(in_channels, out_channels)
        )
        nn.init.xavier_normal_(self.weight)
        if bias is True:
            self.bias = Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias)

    def forward(self, node_ft, adj_mat):
        laplace_mat = get_laplace_mat(adj_mat, type='sym')
        node_state = torch.mm(laplace_mat, node_ft)
        node_state = torch.mm(node_state, self.weight)
        if self.bias is not None:
            node_state = node_state + self.bias

        return node_state


class StructureGAT(nn.Module):

    def __init__(
            self,
            level='aminoacid',
            num_blocks=4,
            hidden_channels=128,
            out_channels=1,
            mid_emb=64,
            num_radial=6,
            num_spherical=2,
            cutoff=10.0,
            max_num_neighbors=32,
            int_emb_layers=3,
            out_layers=2,
            num_pos_emb=16,
            dropout=0,
            data_augment_eachlayer=False,
            euler_noise=False,
    ):
        super(StructureGAT, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_pos_emb = num_pos_emb
        self.data_augment_eachlayer = data_augment_eachlayer
        self.euler_noise = euler_noise
        self.level = level
        self.act = swish

        self.feature0 = d_theta_phi_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature1 = d_angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        if level == 'allatom':
            self.embedding = torch.nn.Linear(num_aa_type + num_bb_embs + num_side_chain_embs, hidden_channels)
        else:
            print('No supported model!')

        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels=hidden_channels,
                    output_channels=hidden_channels,
                    num_radial=num_radial,
                    num_spherical=num_spherical,
                    num_layers=int_emb_layers,
                    mid_emb=mid_emb,
                    act=self.act,
                    num_pos_emb=num_pos_emb,
                    dropout=dropout,
                    level=level
                )
                for _ in range(num_blocks)
            ]
        )

        self.lins_out = torch.nn.ModuleList()
        for _ in range(out_layers - 1):
            self.lins_out.append(Linear(hidden_channels, hidden_channels))
        self.lin_out = Linear(hidden_channels, out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins_out:
            lin.reset_parameters()
        self.lin_out.reset_parameters()

    def pos_emb(self, edge_index, num_pos_emb=16):
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_pos_emb, 2, dtype=torch.float32, device=edge_index.device)
            * -(np.log(10000.0) / num_pos_emb)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def forward(self, batch_data):

        z, pos, batch = torch.squeeze(batch_data.x.long()), batch_data.coords_ca, batch_data.batch
        pos_n = batch_data.coords_n
        pos_c = batch_data.coords_c
        bb_embs = batch_data.bb_embs
        side_chain_embs = batch_data.side_chain_embs
        device = z.device

        if self.level == 'allatom':
            x = torch.cat([torch.squeeze(F.one_hot(z, num_classes=num_aa_type).float()), bb_embs, side_chain_embs],
                          dim=1)
            x = self.embedding(x)
        else:
            print('No supported model!')

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        pos_emb = self.pos_emb(edge_index, self.num_pos_emb)
        j, i = edge_index

        dist = (pos[i] - pos[j]).norm(dim=1)

        num_nodes = len(z)

        # Calculate angles theta and phi.
        refi0 = (i - 1) % num_nodes
        refi1 = (i + 1) % num_nodes

        a = ((pos[j] - pos[i]) * (pos[refi0] - pos[i])).sum(dim=-1)
        b = torch.cross(pos[j] - pos[i], pos[refi0] - pos[i]).norm(dim=-1)
        theta = torch.atan2(b, a)

        plane1 = torch.cross(pos[refi0] - pos[i], pos[refi1] - pos[i])
        plane2 = torch.cross(pos[refi0] - pos[i], pos[j] - pos[i])
        a = (plane1 * plane2).sum(dim=-1)
        b = (torch.cross(plane1, plane2) * (pos[refi0] - pos[i])).sum(dim=-1) / ((pos[refi0] - pos[i]).norm(dim=-1))
        phi = torch.atan2(b, a)

        feature0 = self.feature0(dist, theta, phi)

        if self.level == 'allatom':
            # Calculate Euler angles.
            Or1_x = pos_n[i] - pos[i]
            Or1_z = torch.cross(Or1_x, torch.cross(Or1_x, pos_c[i] - pos[i]))
            Or1_z_length = Or1_z.norm(dim=1) + 1e-7

            Or2_x = pos_n[j] - pos[j]
            Or2_z = torch.cross(Or2_x, torch.cross(Or2_x, pos_c[j] - pos[j]))
            Or2_z_length = Or2_z.norm(dim=1) + 1e-7

            Or1_Or2_N = torch.cross(Or1_z, Or2_z)

            angle1 = torch.atan2((torch.cross(Or1_x, Or1_Or2_N) * Or1_z).sum(dim=-1) / Or1_z_length,
                                 (Or1_x * Or1_Or2_N).sum(dim=-1))
            angle2 = torch.atan2(torch.cross(Or1_z, Or2_z).norm(dim=-1), (Or1_z * Or2_z).sum(dim=-1))
            angle3 = torch.atan2((torch.cross(Or1_Or2_N, Or2_x) * Or2_z).sum(dim=-1) / Or2_z_length,
                                 (Or1_Or2_N * Or2_x).sum(dim=-1))

            if self.euler_noise:
                euler_noise = torch.clip(torch.empty(3, len(angle1)).to(device).normal_(mean=0.0, std=0.025), min=-0.1,
                                         max=0.1)
                angle1 += euler_noise[0]
                angle2 += euler_noise[1]
                angle3 += euler_noise[2]

            feature1 = torch.cat(
                (self.feature1(dist, angle1), self.feature1(dist, angle2), self.feature1(dist, angle3)), 1)


        for interaction_block in self.interaction_blocks:
            if self.data_augment_eachlayer:
                gaussian_noise = torch.clip(torch.empty(x.shape).to(device).normal_(mean=0.0, std=0.025), min=-0.1,
                                            max=0.1)
                x += gaussian_noise
            x = interaction_block(x, feature0, feature1, pos_emb, edge_index, batch)

        y = scatter(x, batch, dim=0)

        return y

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

class MuLAAIP(nn.Module):
    def __init__(self, norm_mode='PN', norm_scale=1):
        super(MuLAAIP, self).__init__()
        struct_hidden = 128
        seq_hidden = 1280
        self.dropout_rate = 0
        self.ab_struct_model = StructureGAT(num_blocks=4, hidden_channels=128, cutoff=10, level='allatom', dropout=0.1)
        self.ag_struct_model = StructureGAT(num_blocks=4, hidden_channels=128, cutoff=10, level='allatom', dropout=0)

        self.fc_ab = nn.Linear(seq_hidden, seq_hidden)
        self.fc_ag = nn.Linear(seq_hidden, seq_hidden)
        self.bn1 = nn.BatchNorm1d(seq_hidden)
        self.bn2 = nn.BatchNorm1d(seq_hidden)
        self.bn3 = nn.BatchNorm1d(struct_hidden)
        self.bn4 = nn.BatchNorm1d(struct_hidden)
        self.bn_struct = nn.BatchNorm1d(2*struct_hidden)
        self.bn_seq = nn.BatchNorm1d(2*seq_hidden)

        self.activation = nn.ReLU()

        hidden_size_combine = struct_hidden + seq_hidden
        self.norm = PairNorm(mode=norm_mode, scale=norm_scale)
        self.ab_gcn1 = GCNConvL(in_channels=seq_hidden, out_channels=seq_hidden)
        self.ag_gcn1 = GCNConvL(in_channels=seq_hidden, out_channels=seq_hidden)

        self.ab_gcn2 = GCNConvL(in_channels=seq_hidden, out_channels=seq_hidden)
        self.ag_gcn2 = GCNConvL(in_channels=seq_hidden, out_channels=seq_hidden)

        self.ab_gcn3 = GCNConvL(in_channels=seq_hidden, out_channels=seq_hidden)
        self.ag_gcn3 = GCNConvL(in_channels=seq_hidden, out_channels=seq_hidden)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size_combine, hidden_size_combine),
            nn.BatchNorm1d(hidden_size_combine),
            nn.ReLU(),
            nn.Linear(hidden_size_combine, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Linear(320, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Linear(160, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, batch_ab, batch_ag, seq_emb_ab, seq_emb_ag):

        struct_ab_emb = self.ab_struct_model(batch_ab)
        struct_ag_emb = self.ag_struct_model(batch_ag)
        combined_embedding = torch.cat((struct_ab_emb, struct_ag_emb), dim=1)

        # bach norm
        s_ab = self.bn3(struct_ab_emb)
        s_ag = self.bn4(struct_ag_emb)
        combined_s = torch.cat((s_ab, s_ag), dim=1)
        combined = combined_s + combined_embedding
        global_struct_feature = self.activation(combined)
        global_struct_feature = self.bn_struct(global_struct_feature)

        ab_in = seq_emb_ab
        ag_in = seq_emb_ag
        ab_0 = self.bn1(ab_in)
        ab_0 = self.activation(ab_0)
        ab_0 = self.fc_ab(ab_0)
        ab_0 = F.dropout(ab_0, p=self.dropout_rate)
        ab_0 = ab_in + ab_0

        w_ab = torch.norm(ab_0, p=2, dim=-1).view(-1, 1)
        w_mat_ab = w_ab * w_ab.t()
        ab_adj = torch.mm(ab_0, ab_0.t()) / w_mat_ab
        ab_1 = self.ab_gcn1(ab_0, ab_adj)
        ab_1 = self.norm(ab_1)
        ab_1 = ab_0 + ab_1

        ab_2 = self.activation(ab_1)
        ab_2 = F.dropout(ab_2, p=self.dropout_rate)
        ab_2 = self.ab_gcn2(ab_2, ab_adj)
        ab_2 = self.norm(ab_2)
        ab_2 = ab_2 + ab_1

        ab_3 = self.activation(ab_2)
        ab_3 = F.dropout(ab_3, p=self.dropout_rate)
        ab_3 = self.ab_gcn3(ab_3, ab_adj)
        ab_3 = self.norm(ab_3)
        ab_3 = ab_3 + ab_2

        ag_0 = self.bn2(ag_in)
        ag_0 = self.activation(ag_0)
        ag_0 = self.fc_ag(ag_0)
        ag_0 = F.dropout(ag_0, p=self.dropout_rate)
        ag_0 = ag_in + ag_0

        w_ag = torch.norm(ag_0, p=2, dim=-1).view(-1, 1)
        w_mat_ag = w_ag * w_ag.t()
        ag_adj = torch.mm(ag_0, ag_0.t()) / w_mat_ag
        ag_1 = self.ag_gcn1(ag_0, ag_adj)
        ag_1 = self.norm(ag_1)
        ag_1 = ag_0 + ag_1

        ag_2 = self.activation(ag_1)
        ag_2 = F.dropout(ag_2, p=self.dropout_rate)
        ag_2 = self.ag_gcn2(ag_2, ag_adj)
        ag_2 = self.norm(ag_2)
        ag_2 = ag_2 + ag_1

        ag_3 = self.activation(ag_2)
        ag_3 = F.dropout(ag_3, p=self.dropout_rate)
        ag_3 = self.ag_gcn3(ag_3, ag_adj)
        ag_3 = self.norm(ag_3)
        ag_3 = ag_3 + ag_2

        x_3 = torch.cat((ab_3, ag_3), dim=1)
        x_2 = torch.cat((ab_2, ag_2), dim=1)
        x_1 = torch.cat((ab_1, ag_1), dim=1)

        global_seq_feature = x_1 + x_2 + x_3 + torch.cat((ab_in, ag_in), dim=1)
        global_seq_feature = self.activation(global_seq_feature)
        global_seq_feature = self.bn_seq(global_seq_feature)

        output = self.mlp(torch.cat((global_seq_feature, global_struct_feature), dim=1))

        return output, ab_adj, ag_adj