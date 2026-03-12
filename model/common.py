from functools import partial
import torch.nn as nn


class BatchNorm1dWrapper(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
    
    def forward(self, x, sample_sizes=None):
        return self.bn(x)


def get_norm(norm_type, num_feats, bn_momentum=0.05, D=-1):
  if norm_type == 'BN':
    return partial(BatchNorm1dWrapper, eps=1e-5, momentum=bn_momentum)
  elif norm_type == 'IN':
    return partial(BatchNorm1dWrapper, eps=1e-5, momentum=bn_momentum)
  else:
    raise ValueError(f'Type {norm_type}, not defined')
