import torch
import torch.nn as nn
from functools import partial
from typing import Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from model.common import get_norm, BatchNorm1dWrapper
from layers import PointConv3d, conv_with_stride
from layers.triplets import build_triplets, voxelize_3d, radius_scaler_for_kernel_size
from layers.metadata import MetaData


def conv3x3x3(in_planes: int, out_planes: int) -> PointConv3d:
    return PointConv3d(in_planes, out_planes, kernel_size=3, bias=False)


def conv1x1x1(in_planes: int, out_planes: int) -> nn.Linear:
    return nn.Linear(in_planes, out_planes, bias=False)


class BasicBlockBase(nn.Module):
  expansion = 1
  NORM_TYPE = 'BN'

  def __init__(self,
               inplanes,
               planes,
               stride=1.0,
               dilation=1,
               downsample=None,
               bn_momentum=0.1,
               D=3):
    super(BasicBlockBase, self).__init__()
    
    norm_layer = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
    
    self.conv1 = conv3x3x3(inplanes, planes)
    self.norm1 = norm_layer(planes)
    self.conv2 = conv3x3x3(planes, planes)
    self.norm2 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=False)
    self.stride = float(stride)
    self.downsample = downsample
    
    if inplanes != planes:
        self.downsample = nn.Sequential(
            conv1x1x1(inplanes, planes),
            norm_layer(planes),
        )

  def forward(self, x: torch.Tensor, m: MetaData) -> Tuple[torch.Tensor, MetaData]:
    identity = x.clone() if self.stride == 1.0 else x.clone()

    x, m = conv_with_stride(self.conv1, x, m, self.stride, receptive_field_scaler=2.5)
    x = self.norm1(x, m.sample_sizes)
    x = self.relu(x)

    if self.stride != 1.0:
        radius_scaler = radius_scaler_for_kernel_size(kernel_size=3, receptive_field_scaler=2.5)
        m.i, m.j, m.k, _ = build_triplets(
            points=m.points,
            sample_inds=m.sample_inds,
            sample_sizes=m.sample_sizes,
            neighbor_radius=m.grid_size * radius_scaler,
            kernel_indexer=partial(voxelize_3d, kernel_size=3),
            radius_scaler=radius_scaler,
        )
    x = self.conv2(x, m.i, m.j, m.k, m.num_points())
    x = self.norm2(x, m.sample_sizes)

    if self.downsample is not None:
        if self.stride == 1.0:
            x_downsample = identity
        else:
            if hasattr(m, 'downsample_indices') and m.downsample_indices is not None:
                x_downsample = identity[m.downsample_indices]
            else:
                x_downsample = identity
        
        for module in self.downsample:
            if isinstance(module, BatchNorm1dWrapper):
                x_downsample = module(x_downsample, sample_sizes=m.sample_sizes)
            elif hasattr(module, '__call__') and 'sample_sizes' in str(module):
                x_downsample = module(x_downsample, m.sample_sizes)
            else:
                x_downsample = module(x_downsample)
        identity = x_downsample

    x = x + identity
    x = self.relu(x)

    return x, m


class BasicBlockBN(BasicBlockBase):
  NORM_TYPE = 'BN'


class BasicBlockIN(BasicBlockBase):
  NORM_TYPE = 'IN'


def get_block(norm_type,
              inplanes,
              planes,
              stride=1.0,
              dilation=1,
              downsample=None,
              bn_momentum=0.1,
              D=3):
  if norm_type == 'BN':
    return BasicBlockBN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
  elif norm_type == 'IN':
    return BasicBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
  else:
    raise ValueError(f'Type {norm_type}, not defined')
