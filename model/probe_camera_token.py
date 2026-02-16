import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from typing import List, Tuple


def get_canonical_idx(v, config):
    canonical = config.model.pose_latent.get(
        'canonical', 'first')

    if canonical == 'first':
        cano_idx = 0
        rel_indices = torch.arange(1, v)
    elif canonical == 'middle':
        cano_idx = v // 2
        rel_indices = torch.cat(
            [torch.arange(cano_idx), torch.arange(cano_idx+1, v)])
    else:
        raise NotImplementedError
    return cano_idx, rel_indices


class MLP(nn.Module):
    def __init__(self, layer_dim: List[int]):
        super().__init__()
        assert len(layer_dim) >= 3, 'MLP must have at least one hidden layer!'
        self.fc = nn.ModuleList()
        for (dim_in, dim_out) in zip(layer_dim, layer_dim[1:]):
            self.fc.append(nn.Linear(dim_in, dim_out))

    def forward(self, x):
        for idx, fc in enumerate(self.fc):
            if idx != len(self.fc)-1:
                x = F.relu(fc(x))
            else:
                x = fc(x)
        return x


class MultiLinear(nn.Module):
    def __init__(self, layer_dim: List[int]):
        super().__init__()
        assert len(
            layer_dim) >= 3, 'MultiLinear must have at least one hidden layer!'
        self.fc = nn.ModuleList()
        for (dim_in, dim_out) in zip(layer_dim, layer_dim[1:]):
            self.fc.append(nn.Linear(dim_in, dim_out))

    def forward(self, x):
        for fc in self.fc:
            x = fc(x)
        return x


def build_probe(probe_type: str, d_in: int, hidden_sizes: List[int], num_classes: int = 2) -> nn.Module:
    probe_type = probe_type.lower()
    if probe_type == "linear":
        return nn.Linear(d_in, num_classes, bias=True)
    elif probe_type in ("mlp", "multilinear"):
        assert hidden_sizes and len(
            hidden_sizes) > 0, "probe_hidden_size must be non-empty for mlp/multilinear"
        layer_dim = [d_in] + list(hidden_sizes) + [num_classes]
        if probe_type == "mlp":
            return MLP(layer_dim=layer_dim)
        else:
            return MultiLinear(layer_dim=layer_dim)
    raise ValueError(f"Unknown probe_type: {probe_type}")


class PoseEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pose_rep = self.config.model.pose_latent.get(
            'representation', '6d')
        print('Pose representation:', self.pose_rep)
        assert self.pose_rep == '6d', f"Only 6d representation is supported for now"
        self.num_pose_element = 6

        def init_weights(m):
            if isinstance(m, nn.Linear):
                # very small weights
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.rel_head = build_probe(config.probe.type,
                                    d_in=config.model.transformer.d*2,
                                    hidden_sizes=config.probe.hidden_sizes,
                                    num_classes=self.num_pose_element+3)
        self.rel_head.apply(init_weights)

    def forward(self, x, v, cano_idx, rel_indices):
        '''
        x: [b*v, d]
        '''
        canonical = self.config.model.pose_latent.get('canonical', 'first')

        x = rearrange(x, '(b v) d -> b v d', v=v)
        b = x.shape[0]
        x_canonical = x[:, cano_idx: cano_idx+1]  # [b,1,d]
        x_rel = x[:, rel_indices]   # [b,v-1,d]

        info_canonical = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0]).reshape(
            1, 1, 9).to(x.device).repeat(b, 1, 1)  # [b,1,9]
        feat_rel = torch.cat(
            [x_canonical.repeat(1, v-1, 1), x_rel], dim=-1)  # [b,v-1,2*d]

        # predict relative pose against canonical for each non-canonical view, [4, v-1, 9] finally
        info_rel = self.rel_head(feat_rel)  # [b,v-1,num_pose_element+3]
        info_all = info_canonical.repeat(
            1, v, 1).to(torch.float32)   # [b,v,num_pose_element+3+2]

        if canonical == 'first':
            info_all[:, 1:, :self.num_pose_element+3] += info_rel
        elif canonical == 'middle':
            info_all[:, rel_indices, :self.num_pose_element+3] += info_rel
        else:
            raise NotImplementedError

        return rearrange(info_all, 'b v d -> (b v) d')
