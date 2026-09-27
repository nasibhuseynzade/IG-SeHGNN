import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# xavier init
def xavier_uniform_(tensor, gain=1.):
    fan_in, fan_out = tensor.size()[-2:]
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return torch.nn.init._no_grad_uniform_(tensor, -a, a)


# semantic fusion
class Transformer(nn.Module):
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none'):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if hasattr(v, 'reset_parameters'):
                v.reset_parameters()
        nn.init.zeros_(self.gamma)

    def forward(self, x, mask=None):
        B, M, C = x.size()
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))

        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3)
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3)

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1)
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h)
        return o.permute(0,2,1,3).reshape((B, M, C)) + x


# per-metapath projection
class LinearPerMetapath(nn.Module):
    def __init__(self, cin, cout, num_metapaths):
        super(LinearPerMetapath, self).__init__()
        self.cin = cin
        self.cout = cout
        self.num_metapaths = num_metapaths

        self.W = nn.Parameter(torch.randn(self.num_metapaths, self.cin, self.cout))
        self.bias = nn.Parameter(torch.zeros(self.num_metapaths, self.cout))

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias.unsqueeze(0)


unfold_nested_list = lambda x: sum(x, [])

# SeHGNN model
class SeHGNN(nn.Module):
    def __init__(self, dataset, nfeat, hidden, nclass, feat_keys, label_feat_keys, tgt_type,
                 dropout, input_drop, att_drop, n_fp_layers, n_task_layers, act,
                 residual=False, data_size=None, drop_metapath=0., num_heads=1):
        super(SeHGNN, self).__init__()
        self.dataset = dataset
        self.feat_keys = sorted(feat_keys)
        self.label_feat_keys = sorted(label_feat_keys)
        self.num_channels = num_channels = len(self.feat_keys) + len(self.label_feat_keys)
        self.tgt_type = tgt_type
        self.residual = residual

        self.input_drop = nn.Dropout(input_drop)

        self.data_size = data_size
        # input embeddings
        self.embeding = nn.ParameterDict({})
        for k, v in data_size.items():
            self.embeding[str(k)] = nn.Parameter(torch.Tensor(v, nfeat))

        if len(self.label_feat_keys):
            self.labels_embeding = nn.ParameterDict({})
            for k in self.label_feat_keys:
                self.labels_embeding[k] = nn.Parameter(torch.Tensor(nclass, nfeat))
        else:
            self.labels_embeding = {}

        # feature projection
        self.feature_projection = nn.Sequential(
            *([LinearPerMetapath(nfeat, hidden, num_channels),
               nn.LayerNorm([num_channels, hidden]),
               nn.PReLU(),
               nn.Dropout(dropout),]
            + unfold_nested_list([[
               LinearPerMetapath(hidden, hidden, num_channels),
               nn.LayerNorm([num_channels, hidden]),
               nn.PReLU(),
               nn.Dropout(dropout),] for _ in range(n_fp_layers - 1)])
            )
        )

        # semantic fusion
        self.semantic_fusion = Transformer(hidden, num_heads=num_heads, att_drop=att_drop, act=act)
        # concatenation layer
        self.fc_after_concat = nn.Linear(num_channels * hidden, hidden)

        if self.residual:
            self.res_fc = nn.Linear(nfeat, hidden)

        if self.dataset != 'IMDB':
            # task mlp
            self.task_mlp = nn.Sequential(
                *([nn.PReLU(),
                   nn.Dropout(dropout),]
                + unfold_nested_list([[
                   nn.Linear(hidden, hidden),
                   nn.BatchNorm1d(hidden, affine=False),
                   nn.PReLU(),
                   nn.Dropout(dropout),] for _ in range(n_task_layers - 1)])
                + [nn.Linear(hidden, nclass),
                   nn.BatchNorm1d(nclass, affine=False, track_running_stats=False)]
                )
            )
        else:
            self.task_mlp = nn.ModuleList(
                [nn.Sequential(
                    nn.PReLU(),
                    nn.Dropout(dropout))]
                + [nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.BatchNorm1d(hidden, affine=False),
                    nn.PReLU(),
                    nn.Dropout(dropout)) for _ in range(n_task_layers - 1)]
                + [nn.Sequential(
                    nn.Linear(hidden, nclass),
                    nn.LayerNorm(nclass, elementwise_affine=False),
                    )]
            )

        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if isinstance(v, nn.ParameterDict):
                for _k, _v in v.items():
                    _v.data.uniform_(-0.5, 0.5)
            elif isinstance(v, nn.ModuleList):
                for block in v:
                    if isinstance(block, nn.Sequential):
                        for layer in block:
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                    elif hasattr(block, 'reset_parameters'):
                        block.reset_parameters()
            elif isinstance(v, nn.Sequential):
                for layer in v:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            elif hasattr(v, 'reset_parameters'):
                v.reset_parameters()

    def forward(self, batch, feature_dict, label_dict={}, mask=None):
        features = {k: self.input_drop(x @ self.embeding[k]) for k, x in feature_dict.items()}

        B = num_node = features[self.tgt_type].shape[0]
        C = self.num_channels
        D = features[self.tgt_type].shape[1]

        labels = {k: self.input_drop(x @ self.labels_embeding[k]) for k, x in label_dict.items()}

        x = [features[k] for k in self.feat_keys] + [labels[k] for k in self.label_feat_keys]
        x = torch.stack(x, dim=1)
        x = self.feature_projection(x)

        x = self.semantic_fusion(x, mask=None).transpose(1,2)

        x = self.fc_after_concat(x.reshape(B, -1))
        if self.residual:
            x = x + self.res_fc(features[self.tgt_type])

        if self.dataset != 'IMDB':
            return self.task_mlp(x)
        else:
            x = self.task_mlp[0](x)
            for i in range(1, len(self.task_mlp)-1):
                x = self.task_mlp[i](x) + x
            x = self.task_mlp[-1](x)
            return x
