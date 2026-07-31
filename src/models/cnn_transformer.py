import uproot
import awkward as ak
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
import os
import numpy as np


import torch



#### CNN encoder for embedding a 2d image as a vector

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 batch_norm=True, activation=nn.ReLU):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=not batch_norm)
        ]

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(activation())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNNEncoder(nn.Module):
    """
    CNN encoder that outputs a fixed-dimensional embedding.

    Parameters
    ----------
    input_channels : int
        Number of image channels.
    channels : list[int]
        Number of output channels for each convolutional stage.
    embedding_dim : int
        Size of final embedding.
    pool_every : int
        Insert MaxPool2d after every `pool_every` conv layers.
    adaptive_size : int
        Output spatial size after adaptive pooling.
    mlp_hidden : int or None
        Hidden dimension in projection head.
    normalize : bool
        L2-normalize the output embedding.
    """

    def __init__(
        self,
        input_channels=3,
        channels=[16, 32, 64,128],
        latent_dim=64,
        proj_dim=32,
        pool_every=1,
        adaptive_size=1,
        projection=True,
        mlp_hidden=None,
        normalize=False,
    ):
        super().__init__()

        layers = []

        in_ch = input_channels

        for i, out_ch in enumerate(channels):
            layers.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch

            if (i + 1) % pool_every == 0:
                layers.append(nn.MaxPool2d(2))

        self.features = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((adaptive_size, adaptive_size))

        feature_dim = channels[-1] * adaptive_size * adaptive_size

        if mlp_hidden is None:
            self.head = nn.Linear(feature_dim, latent_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(feature_dim, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, latent_dim),
            )

        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, 2*latent_dim),
            nn.LayerNorm(2*latent_dim),
            nn.ReLU(),
            nn.Linear(2*latent_dim, proj_dim),
        )

        self.normalize = normalize

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)

        if projection==True:
            z = self.projection_head(x)

        if self.normalize:
            z = nn.functional.normalize(z, dim=1)

        return z





### ViT transformer encoder 

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patch embeddings.

    Input:
        (B, C, H, W)

    Output:
        (B, N, D)

    where
        N = number of patches
        D = embedding dimension
    """

    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=32,
    ):
        super().__init__()

        assert img_size % patch_size == 0

        self.num_patches = (img_size // patch_size) ** 2

        # Conv2d performs patch extraction + linear projection
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)                  # (B,D,H/P,W/P)
        x = x.flatten(2)                  # (B,D,N)
        x = x.transpose(1, 2)             # (B,N,D)
        return x



class ViTEncoder(nn.Module):
    

    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_channels=3,
        feature_dim=64,
        depth=3,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        latent_dim=64,
        proj_dim=32,
        projection=True,
        use_cls_token=True,
        normalize=False,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size,
            patch_size,
            in_channels,
            feature_dim,
        )

        num_patches = self.patch_embed.num_patches

        self.use_cls_token = use_cls_token

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
            num_tokens = num_patches + 1
        else:
            num_tokens = num_patches

        self.pos_embed = nn.Parameter(
            torch.randn(1, num_tokens, feature_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_ratio * feature_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.head = nn.Linear(feature_dim, latent_dim)

        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, 2*latent_dim),
            nn.LayerNorm(2*latent_dim),
            nn.ReLU(),
            nn.Linear(2*latent_dim, proj_dim),
        )

        self.normalize = normalize

    def forward(self, x):

        x = self.patch_embed(x)

        B = x.size(0)

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed

        x = self.transformer(x)

        if self.use_cls_token:
            features = x[:, 0]
        else:
            features = x.mean(dim=1)

        embedding = self.head(features)


        if projection==True:
            z = self.projection_head(embedding)

    
        if self.normalize:
            z = nn.functional.normalize(z, dim=1)

        return z