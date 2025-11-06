# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
import os
import sys
from functools import partial

from .pos_embed import get_2d_sincos_pos_embed


class HybridPatchEmbed(nn.Module):
    """
    CNN-based Hybrid Patch Embedding.
    Dynamically determines the number of patches (num_patches) based on
    the output feature map size.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )
        # Store patch_size in a tuple to match original code usage
        self.patch_size = (patch_size, patch_size)

        # Perform a dummy forward pass to determine num_patches
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_chans, img_size, img_size)
            out = self.conv(dummy_input)
            _, _, H, W = out.shape
            self.num_patches = H * W  # e.g. 14*14 = 196 for 224x224 input

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            Flattened patch embeddings of shape (B, num_patches, embed_dim)
        """
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=True,
        hybrid=False
    ):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # --------------------------------------------------------------------------
        if hybrid:
            self.patch_embed = HybridPatchEmbed(img_size, patch_size, in_chans, embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Fixed sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        # --------------------------------------------------------------------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.hybrid = hybrid  # Track whether the model is using hybrid embedding

        self.initialize_weights()

    def initialize_weights(self):
        # 1) Positional embeddings
        grid_size = int(self.patch_embed.num_patches**0.5)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            grid_size,
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            grid_size,
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # 2) Initialize patch embedding
        if isinstance(self.patch_embed, PatchEmbed):
            # Standard patch embed
            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        elif isinstance(self.patch_embed, HybridPatchEmbed):
            # CNN-based patch embed
            # Optionally apply your desired initialization for conv layers
            for layer in self.patch_embed.conv:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
            # If needed, you could add more specialized init for HybridPatchEmbed here.

        # 3) Initialize tokens
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # 4) Initialize Linear/LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**0.5)
        assert h * w == x.shape[1], "Mismatch in the number of patches and expected shape."

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # (1) Embed patches
        x = self.patch_embed(x)  # [N, num_patches, embed_dim]

        # (2) Add pos embed (excluding cls token)
        x = x + self.pos_embed[:, 1:, :]

        # (3) Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # (4) Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # (5) Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # (1) Project to decoder embedding
        x = self.decoder_embed(x)

        # (2) Prepare mask tokens
        batch_size = x.shape[0]
        seq_len = ids_restore.shape[1] + 1  # +1 for cls
        mask_tokens = self.mask_token.repeat(batch_size, seq_len - x.shape[1], 1)

        # (3) Reinsert masked patches
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # omit cls
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # (4) Add decoder pos embed
        x = x + self.decoder_pos_embed

        # (5) Decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # (6) Predictor projection
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # remove cls token
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0=keep, 1=remove
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L]
        loss = (loss * mask).sum() / mask.sum()  # average over removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.6):
        # 1) Encoder
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        # 2) Decoder
        pred = self.forward_decoder(latent, ids_restore)
        # 3) Loss
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
