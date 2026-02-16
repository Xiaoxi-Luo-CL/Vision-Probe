# Copyright (c) 2025 Hanwen Jiang. Created for the RayZer project.

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from PIL import Image
import traceback

from .transformer import QK_Norm_TransformerBlock, _init_weights_layerwise
from .transformer import init_weights as _init_weights

from utils.data_utils import SplitData
from utils.pe_utils import get_1d_sincos_pos_emb_from_grid, get_2d_sincos_pos_embed
from utils.pose_utils import rot6d2mat


class RayZer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.split_data = SplitData(config)

        # image tokenizer
        self.image_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.config.model.image_tokenizer.patch_size,
                pw=self.config.model.image_tokenizer.patch_size,
            ),
            nn.Linear(
                config.model.image_tokenizer.in_channels
                * (config.model.image_tokenizer.patch_size**2),
                config.model.transformer.d,
                bias=False,
            ),
        )
        self.image_tokenizer.apply(_init_weights)

        # image positional embedding embedder
        self.use_pe_embedding_layer = config.model.get('input_with_pe', True)
        self.pe_embedder = (
            nn.Sequential(
                nn.Linear(
                    config.model.transformer.d,
                    config.model.transformer.d,
                ),
                nn.SiLU(),
                nn.Linear(
                    config.model.transformer.d,
                    config.model.transformer.d,
                ),
            )
            if self.use_pe_embedding_layer
            else nn.Identity()
        )
        self.pe_embedder.apply(_init_weights)

        # pose tokens
        self.cam_code = nn.Parameter(
            torch.randn(
                self.config.model.pose_latent.get('length', 1),
                config.model.transformer.d,
            )
        )
        nn.init.trunc_normal_(self.cam_code, std=0.02)

        # pose pe temporal embedder
        self.temporal_pe_embedder = (
            nn.Sequential(
                nn.Linear(
                    config.model.transformer.d,
                    config.model.transformer.d,
                ),
                nn.SiLU(),
                nn.Linear(
                    config.model.transformer.d,
                    config.model.transformer.d,
                ),
            )
            if self.use_pe_embedding_layer
            else nn.Identity()
        )
        self.temporal_pe_embedder.apply(_init_weights)

        # qk norm settings
        use_qk_norm = config.model.transformer.get("use_qk_norm", False)

        # transformer encoder and init
        self.transformer_encoder = [
            QK_Norm_TransformerBlock(
                config.model.transformer.d, config.model.transformer.d_head, use_qk_norm=use_qk_norm
            )
            for _ in range(config.model.transformer.encoder_n_layer)
        ]
        if config.model.transformer.get("special_init", False):
            if config.model.transformer.get('depth_init', False):
                for idx in range(len(self.transformer_encoder)):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                    self.transformer_encoder[idx].apply(
                        lambda module: _init_weights_layerwise(module, weight_init_std))
            else:
                for idx in range(len(self.transformer_encoder)):
                    weight_init_std = 0.02 / \
                        (2 * config.model.transformer.encoder_n_layer) ** 0.5
                    self.transformer_encoder[idx].apply(
                        lambda module: _init_weights_layerwise(module, weight_init_std))
            self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
        else:
            self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
            self.transformer_encoder.apply(_init_weights)

        # training settings
        if config.inference or config.get("evaluation", False):
            if config.training.get('random_split', False):
                self.random_index = True
            else:
                self.random_index = False
        else:
            self.random_index = config.training.get('random_split', False)
        print('Use random index:', self.random_index)

    def train(self, mode=True):
        # override the train method to keep the fronzon modules in eval mode
        super().train(mode)
        # self.loss_computer.eval()

    def get_overview(self):
        def count_train_params(model): return sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        overview = edict(
            image_tokenizer=count_train_params(self.image_tokenizer),
            pe_embedder=count_train_params(self.pe_embedder),
            temporal_pe_embedder=count_train_params(self.temporal_pe_embedder),
            cam_code=self.cam_code.data.numel(),
            transformer_encoder=count_train_params(self.transformer_encoder)
        )
        return overview

    def forward(self, data):
        '''Split all images into two sets, use one set to get scene representation, use the other to render & train'''
        # batch size:4, input: 16 images, target: 8 images
        input, target, input_idx, target_idx = self.split_data(
            data, random_index=self.random_index)
        # [b, v, c, h, w], range (0,1) to (-1,1)
        image = input.image * 2.0 - 1.0
        b, v_input, c, h, w = image.shape
        # [b, v_all, c, h, w], range (0,1) to (-1,1)
        image_all = data['image'] * 2.0 - 1.0
        v_all = image_all.shape[1]
        device = image.device
        input_idx, target_idx = input_idx.to(device), target_idx.to(device)

        '''se3 pose prediction for all views'''
        # tokenize images, add spatial-temporal p.e.
        img_tokens = self.image_tokenizer(
            image_all)  # [b * v, n, d], [96, 256, 768]
        _, n, d = img_tokens.shape
        if self.use_pe_embedding_layer:
            img_tokens = self.add_sptial_temporal_pe(
                img_tokens, b, v_all, h, w)
        img_tokens = rearrange(
            # [b, v * n, d]
            img_tokens, '(b v) n d -> b (v n) d', b=b, v=v_all)

        # get camera tokens, add temporal p.e.
        # cam_tokens: [b, v_all * n_cam, d]
        cam_tokens = self.get_camera_tokens(b, v_all)
        n_cam = cam_tokens.shape[1] // v_all
        assert n_cam == 1
        # [b, v_all, n_cam, d]
        cam_tokens = rearrange(cam_tokens, 'b (v n) d -> b v n d', v=v_all)
        # [b, v_all * n_cam, d]
        cam_tokens = rearrange(cam_tokens, 'b v n d -> b (v n) d')

        # pose estimation for all views
        all_tokens = torch.cat([cam_tokens, img_tokens], dim=1)
        all_tokens = self.run_encoder(all_tokens)
        # cam_tokens, _ = all_tokens.split([v_all * n_cam, v_all * n], dim=1)
        # get se3 poses and intrinsics
        # cam_tokens = rearrange(cam_tokens, 'b (v n) d -> (b v) n d',
        #    b=b, v=v_all, n=n_cam)[:, 0]  # [b * v_all, d]
        # # [b * v_all, num_pose_element+3+4], rot, 3d trans, 4d fxfycxcy
        # cam_info = self.pose_predictor(cam_tokens, v_all)
        # c2w, fxfycxcy = get_cam_se3(cam_info)  # [b * v_all,4,4], [b * v_all,4]

        result = edict(
            input=input, target=target,
            cam_tokens=cam_tokens,
            # c2w=rearrange(c2w, '(b v) c d -> b v c d', b=b, v=v_all),
            # fxfycxcy=fxfycxcy,
            input_idx=input_idx, target_idx=target_idx,
            n_patch=n, n_cam=n_cam
        )
        return result

    def get_camera_tokens(self, b, v):
        n, d = self.cam_code.shape[-2:]

        # [1, 1, n_cam, d]
        cam_tokens = rearrange(self.cam_code, 'n d -> 1 1 n d')
        cam_tokens = repeat(cam_tokens, '1 1 n d -> 1 v n d',
                            v=v)  # [1, v, n_cam, d]
        cam_tokens = rearrange(
            cam_tokens, '1 v n d -> 1 (v n) d')  # [1, v*n_cam, d]

        # get temporal pe
        img_indices = torch.arange(v).repeat_interleave(n)  # [v*n_cam]
        img_indices = img_indices.to(cam_tokens.device)
        temporal_pe = get_1d_sincos_pos_emb_from_grid(
            embed_dim=d, pos=img_indices,
            device=cam_tokens.device).to(cam_tokens.dtype)  # [v*n_cam, d]
        temporal_pe = temporal_pe.reshape(1, v*n, d)  # [1, v*n_cam, d]
        temporal_pe = self.temporal_pe_embedder(temporal_pe)  # [1, v*n_cam, d]

        # [b, v*n_cam, d]
        return (cam_tokens + temporal_pe).repeat(b, 1, 1)

    def add_sptial_temporal_pe(self, img_tokens, b, v, h_origin, w_origin):
        """
        Adding spatial-temporal pe to input image tokens
        Args:
            img_tokens: shape [b*v, n, d]
        Return:
            image tokens with positional embedding
        """
        patch_size = self.config.model.image_tokenizer.patch_size
        num_h_tokens = h_origin // patch_size
        num_w_tokens = w_origin // patch_size
        assert (num_h_tokens * num_w_tokens) == img_tokens.shape[1]
        bv, n, d = img_tokens.shape

        # get temporal pe
        img_indices = torch.arange(v).repeat_interleave(n)  # [v*n]
        img_indices = img_indices.unsqueeze(
            0).repeat(b, 1).reshape(-1)  # [b*v*n]
        img_indices = img_indices.to(img_tokens.device)
        temporal_pe = get_1d_sincos_pos_emb_from_grid(
            embed_dim=d//2,
            pos=img_indices,
            device=img_tokens.device).to(img_tokens.dtype)  # [b*v*n, d2]
        temporal_pe = temporal_pe.reshape(
            b, v, n, d//2)  # [b,v,n,d2]

        # get spatial pe
        spatial_pe = get_2d_sincos_pos_embed(embed_dim=d//2, grid_size=(
            num_h_tokens, num_w_tokens),  # [n, d3]
            device=img_tokens.device).to(img_tokens.dtype)
        spatial_pe = spatial_pe.reshape(
            1, 1, n, d//2).repeat(b, v, 1, 1)     # [b,v,n,d3]
        # embed pe, [b*v,n,d]
        pe = self.pe_embedder(
            torch.cat([spatial_pe, temporal_pe], dim=-1).reshape(bv, n, d)
        )

        return img_tokens + pe

    def run_layers_encoder(self, start, end):
        def custom_forward(concat_nerf_img_tokens):
            for i in range(start, min(end, len(self.transformer_encoder))):
                concat_nerf_img_tokens = self.transformer_encoder[i](
                    concat_nerf_img_tokens)
            return concat_nerf_img_tokens
        return custom_forward

    def run_encoder(self, all_tokens_encoder):
        checkpoint_every = self.config.training.grad_checkpoint_every
        for i in range(0, len(self.transformer_encoder), checkpoint_every):
            all_tokens_encoder = torch.utils.checkpoint.checkpoint(
                self.run_layers_encoder(i, i+1),
                all_tokens_encoder,
                use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens_encoder = self.run_layers_encoder(
                    i + 1, i + checkpoint_every)(all_tokens_encoder)
        return all_tokens_encoder

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(
                load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name)
                          for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(
                ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None

        self.load_state_dict(checkpoint["model"], strict=False)
        return 0


# utils functions
# def get_cam_se3(cam_info):
#     '''
#     cam_info: [b,num_pose_element+3+4], rot, 3d trans, 4d fxfycxcy
#     '''
#     b, n = cam_info.shape  # (96, 13)

#     if n == 13:
#         rot_6d = cam_info[:, :6]
#         R = rot6d2mat(rot_6d)  # [b,3,3]
#         t = cam_info[:, 6:9].unsqueeze(-1)   # [b,3,1]
#         # normalized by resolution / shift from average, [b,4]
#         fxfycxcy = cam_info[:, 9:]
#     elif n == 11:
#         rot_quat = cam_info[:, :4]
#         R = quat2mat(rot_quat)
#         t = cam_info[:, 4:7].unsqueeze(-1)   # [b,3,1]
#         # normalized by resolution / shift from average, [b,4]
#         fxfycxcy = cam_info[:, 7:]
#     else:
#         raise NotImplementedError

#     Rt = torch.cat([R, t], dim=2)  # [b,3,4]
#     c2w = torch.cat([Rt,  # [b,4,4]
#                      torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device).view(1, 1, 4).repeat(b, 1, 1)], dim=1)
#     return c2w, fxfycxcy


class RayZerHiddenCollector():
    def __init__(self, model: RayZer, layer_to_collect: list):
        self.model = model
        self.layer_to_collect = layer_to_collect
        self.hidden_states = {}
        self.hooks = []

    def _get_hook(self, name):
        def hook(model, input, output):
            self.hidden_states[name] = output.detach().cpu()
        return hook

    def register_hooks(self):
        for i, layer in enumerate(self.model.transformer_encoder):
            hook = layer.register_forward_hook(self._get_hook(f"layer_{i+1}"))
            self.hooks.append(hook)

    def unregister(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def collect_camera_token(self, n_cam, n_patch, v_all):
        '''Collect camera tokens from hidden states after forward pass.'''
        layer_data = {}
        # save camera_tokens and w2c for each view
        for layer, hidden_state in self.hidden_states.items():
            cam_tokens, _ = hidden_state.split(
                [v_all * n_cam, v_all * n_patch], dim=1)
            layer_data[layer] = cam_tokens

        return layer_data
