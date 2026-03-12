# -*- coding: future_fstrings -*-
import torch
import torch.nn as nn
from functools import partial

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from model.common import get_norm
from model.residual_block import get_block
from layers import PointConv3d, conv_with_stride
from layers.upsample import Upsample
from layers.triplets import build_triplets, voxelize_3d, radius_scaler_for_kernel_size
from layers.metadata import MetaData


def conv1x1x1(in_planes: int, out_planes: int) -> nn.Linear:
    return nn.Linear(in_planes, out_planes, bias=False)


class ResUNet2(nn.Module):
  NORM_TYPE = None
  BLOCK_NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]

  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               normalize_feature=None,
               conv1_kernel_size=None,
               D=3,
               voxel_size=0.05):
    super(ResUNet2, self).__init__()
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    self.normalize_feature = normalize_feature
    self.voxel_size = voxel_size
    
    norm_fn = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)
    
    if conv1_kernel_size is None:
        conv1_kernel_size = 5
    
    self.conv1 = PointConv3d(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        bias=False)
    self.norm1 = norm_fn(CHANNELS[1])
    self.relu = nn.ReLU(inplace=False)

    self.block1 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], stride=1.0, bn_momentum=bn_momentum, D=D)

    self.conv2 = PointConv3d(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        bias=False)
    self.norm2 = norm_fn(CHANNELS[2])

    self.block2 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], stride=1.0, bn_momentum=bn_momentum, D=D)

    self.conv3 = PointConv3d(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        bias=False)
    self.norm3 = norm_fn(CHANNELS[3])

    self.block3 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], stride=1.0, bn_momentum=bn_momentum, D=D)

    self.conv4 = PointConv3d(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        bias=False)
    self.norm4 = norm_fn(CHANNELS[4])

    self.block4 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], stride=1.0, bn_momentum=bn_momentum, D=D)

    self.conv4_tr = Upsample(
        in_channels=CHANNELS[4],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        bias=False,
        receptive_field_scaler=1.0)
    self.norm4_tr = norm_fn(TR_CHANNELS[4])

    self.block4_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], stride=1.0, bn_momentum=bn_momentum, D=D)

    self.conv3_tr = Upsample(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        bias=False,
        receptive_field_scaler=1.0)
    self.norm3_tr = norm_fn(TR_CHANNELS[3])

    self.block3_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], stride=1.0, bn_momentum=bn_momentum, D=D)

    self.conv2_tr = Upsample(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        bias=False,
        receptive_field_scaler=1.0)
    self.norm2_tr = norm_fn(TR_CHANNELS[2])

    self.block2_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], stride=1.0, bn_momentum=bn_momentum, D=D)

    self.conv1_tr = PointConv3d(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        bias=False)

    self.final = PointConv3d(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        bias=True)

  def forward(self, input_dict):
    """
    Forward pass with pointcnnpp format.
    
    Args:
        input_dict: Dictionary containing:
            - 'feat': [N, in_channels] input features
            - 'coord': [N, 3] point coordinates
            - 'offset': [B] batch offsets
            - 'grid_size' or 'voxel_size': grid size (optional, defaults to self.voxel_size)
    
    Returns:
        [N, out_channels] output features
    """
    feat = input_dict["feat"]
    offset = input_dict["offset"]
    
    if "coord" in input_dict:
        coord = input_dict["coord"]
    elif "grid_coord" in input_dict:
        coord = input_dict["grid_coord"]
    else:
        raise ValueError("input_dict must contain either 'coord' or 'grid_coord'")
    
    # Ensure coord is float32
    if coord.dtype != torch.float32:
        coord = coord.float()
    
    # Ensure feat is float32
    if feat.dtype != torch.float32:
        feat = feat.float()
    
    # Get grid_size
    grid_size = input_dict.get("grid_size", input_dict.get("voxel_size", self.voxel_size))
    
    # Ensure offset is on the same device as coord
    if offset.device != coord.device:
        offset = offset.to(coord.device)
    
    # Create sample_sizes from offset
    sample_sizes = torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=offset.dtype))
    
    # Create sample_inds - ensure it's on the same device as coord
    sample_inds = torch.repeat_interleave(
        torch.arange(0, sample_sizes.numel(), device=coord.device, dtype=torch.long),
        sample_sizes,
    )
    
    # Ensure sample_sizes is also on the correct device
    if sample_sizes.device != coord.device:
        sample_sizes = sample_sizes.to(coord.device)
    
    # Create MetaData
    m = MetaData(
        points=coord,
        sample_inds=sample_inds,
        sample_sizes=sample_sizes,
        grid_size=grid_size,
    )
    
    x = feat
    
    # Encoder
    x, m = conv_with_stride(self.conv1, x, m, 1.0, receptive_field_scaler=1.0)
    m.dirty_triplets()
    x = self.norm1(x, m.sample_sizes)
    x = self.relu(x)
    
    out_s1, m = self.block1(x, m)
    out = self.relu(out_s1)
    
    # Save for skip connection (no clone: out is reassigned later, not mutated in place)
    down_outputs = [out]
    # Save MetaData snapshot (no clone: handle_stride_and_build_triplets reassigns m.points/etc, does not mutate in place)
    down_metadatas = [MetaData(
        points=m.points,
        sample_inds=m.sample_inds,
        sample_sizes=m.sample_sizes,
        grid_size=m.grid_size,
        parent=m.parent,
    )]
    
    # Downsample with stride 2
    x, m = conv_with_stride(self.conv2, out, m, 2.0, receptive_field_scaler=1.0)
    m.dirty_triplets()
    x = self.norm2(x, m.sample_sizes)
    x = self.relu(x)
    out_s2, m = self.block2(x, m)
    out = self.relu(out_s2)
    
    down_outputs.append(out)
    down_metadatas.append(MetaData(
        points=m.points,
        sample_inds=m.sample_inds,
        sample_sizes=m.sample_sizes,
        grid_size=m.grid_size,
        parent=down_metadatas[0] if len(down_metadatas) > 0 else m.parent,
    ))
    
    # Downsample with stride 2
    x, m = conv_with_stride(self.conv3, out, m, 2.0, receptive_field_scaler=1.0)
    m.dirty_triplets()
    x = self.norm3(x, m.sample_sizes)
    x = self.relu(x)
    out_s4, m = self.block3(x, m)
    out = self.relu(out_s4)
    
    down_outputs.append(out)
    down_metadatas.append(MetaData(
        points=m.points,
        sample_inds=m.sample_inds,
        sample_sizes=m.sample_sizes,
        grid_size=m.grid_size,
        parent=down_metadatas[-1] if len(down_metadatas) > 0 else m.parent,
    ))
    
    # Downsample with stride 2
    x, m = conv_with_stride(self.conv4, out, m, 2.0, receptive_field_scaler=1.0)
    # After conv_with_stride, m.parent should point to out_s4 metadata
    m.dirty_triplets()
    x = self.norm4(x, m.sample_sizes)
    x = self.relu(x)
    out_s8, m = self.block4(x, m)
    out = self.relu(out_s8)
    
    # Decoder
    # Stage 4: upsample (from out_s8 to out_s4 resolution)
    # Follow unet_pointcnnpp.py: upsample first, then norm/relu, then blocks, then concat
    # The Upsample will automatically restore points from m.parent if it exists
    # After conv_with_stride, m.parent should point to out_s4 metadata, but block4 might have modified it
    # Force restore parent from saved metadata to ensure correctness
    # down_outputs: [out_s1, out_s2, out_s4], so down_outputs[-1] is out_s4
    m.parent = down_metadatas[-1]  # Force set parent to out_s4 metadata
    
    out, m = self.conv4_tr(out, m)
    out = self.norm4_tr(out, m.sample_sizes)
    out = self.relu(out)
    out, m = self.block4_tr(out, m)
    
    # Concatenate with skip connection (down_outputs[-1] is out_s4)
    out = torch.cat([out, down_outputs[-1]], dim=1)
    
    # Stage 3: upsample (from out_s4 to out_s2 resolution)
    # After previous upsample, m.parent should point to out_s2 metadata, but block might have modified it
    # Force restore parent from saved metadata to ensure correctness
    # down_outputs: [out_s1, out_s2, out_s4], so down_outputs[-2] is out_s2
    m.parent = down_metadatas[-2]  # Force set parent to out_s2 metadata
    
    out, m = self.conv3_tr(out, m)
    out = self.norm3_tr(out, m.sample_sizes)
    out = self.relu(out)
    out, m = self.block3_tr(out, m)
    
    # Concatenate with skip connection (down_outputs[-2] is out_s2)
    out = torch.cat([out, down_outputs[-2]], dim=1)
    
    # Stage 2: upsample (from out_s2 to out_s1 resolution)
    # After previous upsample, m.parent should point to out_s1 metadata, but block might have modified it
    # Force restore parent from saved metadata to ensure correctness
    m.parent = down_metadatas[0]  # Force set parent to out_s1 metadata
    
    out, m = self.conv2_tr(out, m)
    out = self.norm2_tr(out, m.sample_sizes)
    out = self.relu(out)
    out, m = self.block2_tr(out, m)
    
    # Concatenate with skip connection (down_outputs[0] is out_s1)
    out = torch.cat([out, down_outputs[0]], dim=1)
    
    # Final conv
    m.dirty_triplets()
    radius_scaler = radius_scaler_for_kernel_size(kernel_size=1, receptive_field_scaler=1.0)
    neighbor_radius = m.grid_size * radius_scaler
    m.i, m.j, m.k, _ = build_triplets(
        points=m.points,
        sample_inds=m.sample_inds,
        sample_sizes=m.sample_sizes,
        neighbor_radius=neighbor_radius,
        kernel_indexer=partial(voxelize_3d, kernel_size=1),
        radius_scaler=radius_scaler,
    )
    out = self.conv1_tr(out, m.i, m.j, m.k, m.num_points())
    out = self.relu(out)
    
    # Final output
    m.dirty_triplets()
    radius_scaler = radius_scaler_for_kernel_size(kernel_size=1, receptive_field_scaler=1.0)
    neighbor_radius = m.grid_size * radius_scaler
    m.i, m.j, m.k, _ = build_triplets(
        points=m.points,
        sample_inds=m.sample_inds,
        sample_sizes=m.sample_sizes,
        neighbor_radius=neighbor_radius,
        kernel_indexer=partial(voxelize_3d, kernel_size=1),
        radius_scaler=radius_scaler,
    )
    out = self.final(out, m.i, m.j, m.k, m.num_points())

    if self.normalize_feature:
      out = out / torch.norm(out, p=2, dim=1, keepdim=True).clamp(min=1e-8)
    
    return out


class ResUNetBN2(ResUNet2):
  NORM_TYPE = 'BN'


class ResUNetBN2B(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2C(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]


class ResUNetBN2D(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2E(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 128, 128, 128, 256]
  TR_CHANNELS = [None, 64, 128, 128, 128]


class ResUNetIN2(ResUNet2):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2B(ResUNetBN2B):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2C(ResUNetBN2C):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2D(ResUNetBN2D):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2E(ResUNetBN2E):
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'IN'
