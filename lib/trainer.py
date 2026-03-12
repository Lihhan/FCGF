import os
import os.path as osp
import gc
import logging
import numpy as np
import json

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from easydict import EasyDict

from model import load_model
import util.transform_estimation as te
from lib.metrics import pdist, corr_dist
from lib.timer import AverageMeter
from lib.eval import find_nn_gpu

from util.file import ensure_dir
from util.misc import _hash

# Add Open3D for RANSAC
try:
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    logging.warning("Open3D not available, RANSAC will not be used")


def run_ransac_registration(xyz0, xyz1, feat0, feat1, voxel_size):
    if not HAS_OPEN3D:
        logging.warning("Open3D not available, falling back to est_quad_linear_robust")
        return None

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(xyz0)
    pcd0_features = o3d.pipelines.registration.Feature()
    pcd0_features.data = feat0.T.astype(np.float32)
    
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(xyz1)
    pcd1_features = o3d.pipelines.registration.Feature()
    pcd1_features.data = feat1.T.astype(np.float32)

    distance_threshold = voxel_size * 1.5
    
    try:
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd0, pcd1, pcd0_features, pcd1_features,
            mutual_filter=False,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 
            ransac_n=4, 
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], 
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999)
        )
        return result_ransac.transformation.astype(np.float32)
    except AttributeError:
        try:
            result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
                pcd0, pcd1, pcd0_features, pcd1_features,
                mutual_filter=False,
                max_correspondence_distance=distance_threshold,
                estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[
                    o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ],
                criteria=o3d.registration.RANSACConvergenceCriteria(50000, 0.999)
            )
            return result_ransac.transformation.astype(np.float32)
        except Exception as e:
            logging.warning(f"RANSAC failed: {e}, falling back to est_quad_linear_robust")
            return None


def extract_features(model_output):
    if hasattr(model_output, 'F'):
        return model_output.F
    
    if hasattr(model_output, 'feat'):
        return model_output.feat
    
    if isinstance(model_output, dict) and 'feat' in model_output:
        return model_output['feat']
    
    if hasattr(model_output, 'shape'):
        return model_output
    
    raise ValueError(f"Cannot extract features from type: {type(model_output)}")


def extract_coords(model_output, fallback_coords=None):
    if isinstance(model_output, dict):
        if 'coord' in model_output:
            coord = model_output['coord']
            if hasattr(coord, 'cpu'):
                return coord.cpu()
            else:
                return coord
    
    if hasattr(type(model_output), 'coord'):
        coord = model_output.coord
        if hasattr(coord, 'cpu'):
            return coord.cpu()
        else:
            return coord
    
    if fallback_coords is not None:
        if isinstance(fallback_coords, torch.Tensor):
            return fallback_coords.cpu() if fallback_coords.is_cuda else fallback_coords
        else:
            return fallback_coords
    
    return None

# MinkowskiEngine removed, using pointcnnpp instead


class AlignmentTrainer:

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

    # Model initialization
    Model = load_model(config.model)
    model = Model(
        num_feats,
        config.model_n_out,
        bn_momentum=config.bn_momentum,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        D=3)

    if config.weights:
      with torch.serialization.safe_globals([EasyDict]):
        checkpoint = torch.load(config.weights, weights_only=False)
      model.load_state_dict(checkpoint['state_dict'])

    logging.info(model)

    self.config = config
    self.model = model
    self.max_epoch = config.max_epoch
    self.save_freq = config.save_freq_epoch
    self.val_max_iter = config.val_max_iter
    self.val_epoch_freq = config.val_epoch_freq

    self.best_val_metric = config.best_val_metric
    self.best_val_epoch = -np.inf
    self.best_val = -np.inf

    if config.use_gpu and not torch.cuda.is_available():
      logging.warning('Warning: There\'s no CUDA support on this machine, '
                      'training is performed on CPU.')
      raise ValueError('GPU not available, but cuda flag set')

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.optimizer = getattr(optim, config.optimizer)(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

    self.start_epoch = 1
    self.checkpoint_dir = config.out_dir

    ensure_dir(self.checkpoint_dir)
    json.dump(
        config,
        open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
        indent=4,
        sort_keys=False)

    self.iter_size = config.iter_size
    self.batch_size = data_loader.batch_size
    self.data_loader = data_loader
    self.val_data_loader = val_data_loader

    self.test_valid = True if self.val_data_loader is not None else False
    self.log_step = int(np.sqrt(self.config.batch_size))
    self.model = self.model.to(self.device)
    self.writer = SummaryWriter(logdir=config.out_dir)

    if config.resume is not None:
      if osp.isfile(config.resume):
        logging.info("=> loading checkpoint '{}'".format(config.resume))
        with torch.serialization.safe_globals([EasyDict]):
          state = torch.load(config.resume, weights_only=False)
        self.start_epoch = state['epoch']
        model.load_state_dict(state['state_dict'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.optimizer.load_state_dict(state['optimizer'])

        if 'best_val' in state.keys():
          self.best_val = state['best_val']
          self.best_val_epoch = state['best_val_epoch']
          self.best_val_metric = state['best_val_metric']
      else:
        raise ValueError(f"=> no checkpoint found at '{config.resume}'")

  def train(self):
    """
    Full training logic
    """
    # Baseline random feature performance
    if self.test_valid:
      with torch.no_grad():
        val_dict = self._valid_epoch()

      for k, v in val_dict.items():
        self.writer.add_scalar(f'val/{k}', v, 0)

    for epoch in range(self.start_epoch, self.max_epoch + 1):
      lr = self.scheduler.get_lr()
      logging.info(f" Epoch: {epoch}, LR: {lr}")
      self._train_epoch(epoch)
      self._save_checkpoint(epoch)
      self.scheduler.step()

      if self.test_valid and epoch % self.val_epoch_freq == 0:
        with torch.no_grad():
          val_dict = self._valid_epoch()

        for k, v in val_dict.items():
          self.writer.add_scalar(f'val/{k}', v, epoch)
        if self.best_val < val_dict[self.best_val_metric]:
          logging.info(
              f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric]}'
          )
          self.best_val = val_dict[self.best_val_metric]
          self.best_val_epoch = epoch
          self._save_checkpoint(epoch, 'best_val_checkpoint')
        else:
          logging.info(
              f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
          )

  def _save_checkpoint(self, epoch, filename='checkpoint'):
    state = {
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'config': self.config,
        'best_val': self.best_val,
        'best_val_epoch': self.best_val_epoch,
        'best_val_metric': self.best_val_metric
    }
    filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
    logging.info("Saving checkpoint: {} ...".format(filename))
    torch.save(state, filename)


class ContrastiveLossTrainer(AlignmentTrainer):

  def __init__(
      self,
      config,
      data_loader,
      val_data_loader=None,
  ):
    if val_data_loader is not None:
      assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
    AlignmentTrainer.__init__(self, config, data_loader, val_data_loader)
    self.neg_thresh = config.neg_thresh
    self.pos_thresh = config.pos_thresh
    self.neg_weight = config.neg_weight

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    
    if isinstance(R, torch.Tensor):
      R_T = R.t()
    else:
      R_T = R.T
    
    return pts @ R_T + T

  def generate_rand_negative_pairs(self, positive_pairs, hash_seed, N0, N1, N_neg=0):
    """
    Generate random negative pairs
    """
    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)
    if N_neg < 1:
      N_neg = positive_pairs.shape[0] * 2
    pos_keys = _hash(positive_pairs, hash_seed)

    neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
        np.int64)
    neg_keys = _hash(neg_pairs, hash_seed)
    mask = np.isin(neg_keys, pos_keys, assume_unique=False)
    return neg_pairs[np.logical_not(mask)]

  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0

    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()

    iter_size = self.iter_size
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)

    # Main training
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

      for iter_idx in range(iter_size):
        input_dict = next(data_loader_iter)

        # pairs consist of (xyz1 index, xyz0 index)
        # Convert to pointcnnpp format
        sinput0_dict = {
            'feat': input_dict['sinput0_F'].to(self.device),
            'coord': input_dict['sinput0_C'].to(self.device),
            'offset': input_dict.get('sinput0_offset', torch.tensor([0, input_dict['sinput0_F'].shape[0]], dtype=torch.long, device=self.device)),
            'grid_size': self.config.voxel_size,
        }
        model_output0 = self.model(sinput0_dict)
        F0 = extract_features(model_output0)

        sinput1_dict = {
            'feat': input_dict['sinput1_F'].to(self.device),
            'coord': input_dict['sinput1_C'].to(self.device),
            'offset': input_dict.get('sinput1_offset', torch.tensor([0, input_dict['sinput1_F'].shape[0]], dtype=torch.long, device=self.device)),
            'grid_size': self.config.voxel_size,
        }
        model_output1 = self.model(sinput1_dict)
        F1 = extract_features(model_output1)

        # Calculate number of points from offset
        N0 = sinput0_dict['offset'][-1].item() if sinput0_dict['offset'].numel() > 0 else sinput0_dict['feat'].shape[0]
        N1 = sinput1_dict['offset'][-1].item() if sinput1_dict['offset'].numel() > 0 else sinput1_dict['feat'].shape[0]

        pos_pairs = input_dict['correspondences']
        neg_pairs = self.generate_rand_negative_pairs(pos_pairs, max(N0, N1), N0, N1)
        pos_pairs = pos_pairs.long().to(self.device)
        neg_pairs = torch.from_numpy(neg_pairs).long().to(self.device)

        neg0 = F0.index_select(0, neg_pairs[:, 0])
        neg1 = F1.index_select(0, neg_pairs[:, 1])
        pos0 = F0.index_select(0, pos_pairs[:, 0])
        pos1 = F1.index_select(0, pos_pairs[:, 1])

        # Positive loss
        pos_loss = (pos0 - pos1).pow(2).sum(1)

        # Negative loss
        neg_loss = F.relu(self.neg_thresh -
                          ((neg0 - neg1).pow(2).sum(1) + 1e-4).sqrt()).pow(2)

        pos_loss_mean = pos_loss.mean() / iter_size
        neg_loss_mean = neg_loss.mean() / iter_size

        # Weighted loss
        loss = pos_loss_mean + self.neg_weight * neg_loss_mean
        loss.backward(
        )  # To accumulate gradient, zero gradients only at the begining of iter_size
        batch_loss += loss.item()
        batch_pos_loss += pos_loss_mean.item()
        batch_neg_loss += neg_loss_mean.item()

      self.optimizer.step()

      if curr_iter % 100 == 0:
        torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0

      # Print logs
      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)  
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, batch_pos_loss, batch_neg_loss))

  def _valid_epoch(self):
    self.model.eval()
    self.val_data_loader.dataset.reset_seed(0)
    np.random.seed(0)
    num_data = 0
    hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = AverageMeter(
    ), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    tot_num_data = len(self.val_data_loader.dataset)
    if self.val_max_iter > 0:
      tot_num_data = min(self.val_max_iter, tot_num_data)
    data_loader_iter = self.val_data_loader.__iter__()

    for batch_idx in range(tot_num_data):
      input_dict = next(data_loader_iter)
      
      if 'pcd0' in input_dict and 'pcd1' in input_dict:
          offset0 = torch.cumsum(torch.tensor([0] + [len(input_dict['pcd0'])], dtype=torch.long), dim=0).to(self.device)
          offset1 = torch.cumsum(torch.tensor([0] + [len(input_dict['pcd1'])], dtype=torch.long), dim=0).to(self.device)
          
          point0_dict = {
              'coord': input_dict['pcd0'].to(self.device),
              'feat': input_dict['sinput0_F'].to(self.device),
              'offset': offset0,
              'grid_size': self.config.voxel_size,
          }
          
          point1_dict = {
              'coord': input_dict['pcd1'].to(self.device),
              'feat': input_dict['sinput1_F'].to(self.device),
              'offset': offset1,
              'grid_size': self.config.voxel_size,
          }
          
          model_output0 = self.model(point0_dict)
          F0 = extract_features(model_output0)
          xyz0_model = extract_coords(model_output0, input_dict['pcd0'])
          if xyz0_model is None:
              xyz0_model = input_dict['pcd0']
          
          model_output1 = self.model(point1_dict)
          F1 = extract_features(model_output1)
          xyz1_model = extract_coords(model_output1, input_dict['pcd1'])
          if xyz1_model is None:
              xyz1_model = input_dict['pcd1']
      else:
          offset0 = torch.tensor([0, input_dict['sinput0_F'].shape[0]], dtype=torch.long, device=self.device)
          offset1 = torch.tensor([0, input_dict['sinput1_F'].shape[0]], dtype=torch.long, device=self.device)
          
          sinput0_dict = {
              'feat': input_dict['sinput0_F'].to(self.device),
              'coord': input_dict['sinput0_C'].to(self.device),
              'offset': offset0,
              'grid_size': self.config.voxel_size,
          }
          model_output0 = self.model(sinput0_dict)
          F0 = extract_features(model_output0)

          sinput1_dict = {
              'feat': input_dict['sinput1_F'].to(self.device),
              'coord': input_dict['sinput1_C'].to(self.device),
              'offset': offset1,
              'grid_size': self.config.voxel_size,
          }
          model_output1 = self.model(sinput1_dict)
          F1 = extract_features(model_output1)

          if 'pcd0' in input_dict:
              xyz0_model = input_dict['pcd0']
          elif 'pcd0_original' in input_dict:
              xyz0_model = input_dict['pcd0_original']
          else:
              logging.warning("No original point cloud coordinates found, using quantized coordinates (may affect hit_ratio accuracy)")
              if isinstance(input_dict['sinput0_C'], torch.Tensor):
                  xyz0_model = input_dict['sinput0_C'].float() * self.config.voxel_size
              else:
                  xyz0_model = np.array(input_dict['sinput0_C'], dtype=np.float32) * self.config.voxel_size
          
          if 'pcd1' in input_dict:
              xyz1_model = input_dict['pcd1']
          elif 'pcd1_original' in input_dict:
              xyz1_model = input_dict['pcd1_original']
          else:
              logging.warning("No original point cloud coordinates found, using quantized coordinates (may affect hit_ratio accuracy)")
              if isinstance(input_dict['sinput1_C'], torch.Tensor):
                  xyz1_model = input_dict['sinput1_C'].float() * self.config.voxel_size
              else:
                  xyz1_model = np.array(input_dict['sinput1_C'], dtype=np.float32) * self.config.voxel_size

      if isinstance(xyz0_model, torch.Tensor):
          xyz0 = xyz0_model.cpu().numpy()
      elif xyz0_model is None:
          raise ValueError("xyz0_model is None, cannot proceed")
      else:
          xyz0 = np.array(xyz0_model, dtype=np.float32)
      
      if isinstance(xyz1_model, torch.Tensor):
          xyz1 = xyz1_model.cpu().numpy()
      elif xyz1_model is None:
          raise ValueError("xyz1_model is None, cannot proceed")
      else:
          xyz1 = np.array(xyz1_model, dtype=np.float32)
      
      T_gt = input_dict['T_gt']
      if isinstance(T_gt, torch.Tensor):
        T_gt = T_gt.cpu().numpy()
      
      xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=1000)
      
      if len(xyz0_corr) < 4:
        logging.warning(f"find_corr returned only {len(xyz0_corr)} matches, skipping this sample")
        num_data += 1
        continue
      
      if isinstance(xyz0_corr, np.ndarray):
        xyz0_corr = torch.from_numpy(xyz0_corr).float()
      if isinstance(xyz1_corr, np.ndarray):
        xyz1_corr = torch.from_numpy(xyz1_corr).float()
      
      T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)
      if isinstance(T_est, torch.Tensor):
        T_est = T_est.cpu().numpy()

      loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
      loss_meter.update(loss)

      rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
      rte_meter.update(rte)
      R_est = T_est[:3, :3]
      R_gt = T_gt[:3, :3]
      rre = np.arccos(np.clip((np.trace(R_est.T @ R_gt) - 1) / 2, -1, 1))
      if not np.isnan(rre):
        rre_meter.update(rre)

      hit_ratio = self.evaluate_hit_ratio(
          xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
      hit_ratio_meter.update(hit_ratio)
      feat_match_ratio.update(hit_ratio > 0.05)

      num_data += 1
      if batch_idx % 100 == 0:
        torch.cuda.empty_cache()

      if batch_idx % 100 == 0:
        logging.info(' '.join([
            f"Validation iter {num_data} / {tot_num_data} :",
            f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
            f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
        ]))

    logging.info(' '.join([
        f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
        f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
    ]))
    return {
        "loss": loss_meter.avg,
        "rre": rre_meter.avg,
        "rte": rte_meter.avg,
        'feat_match_ratio': feat_match_ratio.avg,
        'hit_ratio': hit_ratio_meter.avg
    }

  def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1):
    nn_result = find_nn_gpu(F0, F1, nn_max_n=self.config.nn_max_n, return_distance=True, dist_type='SquareL2')
    
    if isinstance(nn_result, tuple):
      nn_inds_01, nn_dists = nn_result
    else:
      nn_inds_01 = nn_result
      nn_dists = None
    
    if isinstance(nn_inds_01, torch.Tensor):
      nn_inds_01 = nn_inds_01.numpy()
    if nn_dists is not None and isinstance(nn_dists, torch.Tensor):
      nn_dists = nn_dists.numpy()
    
    N0 = F0.shape[0] if hasattr(F0, 'shape') else len(F0)
    N1 = F1.shape[0] if hasattr(F1, 'shape') else len(F1)
    valid_mask = (nn_inds_01 >= 0) & (nn_inds_01 < N1)
    
    if valid_mask.sum() == 0:
      logging.warning(f"find_corr: no valid matches found (N0={N0}, N1={N1}); returning empty matches")
      if isinstance(xyz0, torch.Tensor):
        xyz0 = xyz0.cpu().numpy()
      if isinstance(xyz1, torch.Tensor):
        xyz1 = xyz1.cpu().numpy()
      return np.empty((0, 3), dtype=xyz0.dtype), np.empty((0, 3), dtype=xyz1.dtype)
    
    nn_inds = nn_inds_01[valid_mask]
    valid_inds0 = np.where(valid_mask)[0]
    
    if subsample_size > 0 and len(valid_inds0) > subsample_size:
      if nn_dists is not None:
        valid_dists = nn_dists[valid_mask]
        if isinstance(valid_dists, torch.Tensor):
          valid_dists = valid_dists.squeeze().cpu().numpy()
        else:
          valid_dists = np.asarray(valid_dists).squeeze()
        
        if valid_dists.ndim > 1:
          valid_dists = valid_dists.flatten()
        
        epsilon = 1e-8
        dist_min, dist_max = valid_dists.min(), valid_dists.max()
        if dist_max > dist_min:
          normalized_dists = (valid_dists - dist_min) / (dist_max - dist_min + epsilon)
        else:
          normalized_dists = np.ones_like(valid_dists)
        conf = 1.0 / (normalized_dists + epsilon)
        prob = conf / conf.sum()
        
        idx = np.arange(len(valid_inds0))
        sampled_idx = np.random.choice(idx, size=subsample_size, replace=False, p=prob)
        valid_inds0 = valid_inds0[sampled_idx]
        nn_inds = nn_inds[sampled_idx]
      else:
        idx = np.arange(len(valid_inds0))
        sampled_idx = np.random.choice(idx, size=subsample_size, replace=False)
        valid_inds0 = valid_inds0[sampled_idx]
        nn_inds = nn_inds[sampled_idx]
    
    if isinstance(xyz0, torch.Tensor):
      xyz0 = xyz0.cpu().numpy()
    if isinstance(xyz1, torch.Tensor):
      xyz1 = xyz1.cpu().numpy()
    
    return xyz0[valid_inds0], xyz1[nn_inds]

  def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
    if isinstance(xyz0, torch.Tensor):
      xyz0 = xyz0.cpu().numpy()
    if isinstance(xyz1, torch.Tensor):
      xyz1 = xyz1.cpu().numpy()
    if isinstance(T_gth, torch.Tensor):
      T_gth = T_gth.cpu().numpy()
    
    xyz0 = self.apply_transform(xyz0, T_gth)
    dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
    return (dist < thresh).astype(np.float32).mean()

  def evaluate_chamfer_distance(self, xyz0, xyz1, T_est, max_points=500):
    """
    计算 Chamfer Distance（倒角距离）
    
    Args:
        xyz0: [N0, 3] 点云0
        xyz1: [N1, 3] 点云1  
        T_est: [4, 4] 估计的变换矩阵
        max_points: int - 最大采样点数（防止OOM）
    
    Returns:
        cd: float - 对称 Chamfer Distance
    """
    xyz1_transformed = self.apply_transform(xyz1, T_est)

    if torch.is_tensor(xyz0):
      xyz0 = xyz0.cpu().numpy()
    if torch.is_tensor(xyz1_transformed):
      xyz1_transformed = xyz1_transformed.cpu().numpy()

    N0, N1 = xyz0.shape[0], xyz1_transformed.shape[0]
    if N0 > max_points:
      inds0 = np.random.choice(N0, max_points, replace=False)
      xyz0 = xyz0[inds0]
    if N1 > max_points:
      inds1 = np.random.choice(N1, max_points, replace=False)
      xyz1_transformed = xyz1_transformed[inds1]

    # xyz0 → xyz1_transformed
    dist_01 = np.sqrt(((xyz0[:, None, :] - xyz1_transformed[None, :, :])**2).sum(2))  # [N0, N1]
    min_dist_01 = dist_01.min(axis=1)  # [N0]
    
    # xyz1_transformed → xyz0
    min_dist_10 = dist_01.min(axis=0)  # [N1]

    cd = (min_dist_01.mean() + min_dist_10.mean()) / 2.0
    
    return cd


class HardestContrastiveLossTrainer(ContrastiveLossTrainer):

  def contrastive_hardest_negative_loss(self,
                                        F0,
                                        F1,
                                        positive_pairs,
                                        num_pos=5192,
                                        num_hn_samples=2048,
                                        thresh=None):
    N0, N1 = F0.shape[0] if hasattr(F0, 'shape') else len(F0), F1.shape[0] if hasattr(F1, 'shape') else len(F1)
    N_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if N_pos_pairs > num_pos:
      pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs
    
    if not isinstance(sample_pos_pairs, np.ndarray):
      sample_pos_pairs = np.array(sample_pos_pairs, dtype=np.int64)

    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = torch.from_numpy(sample_pos_pairs[:, 0]).long().to(F0.device)
    pos_ind1 = torch.from_numpy(sample_pos_pairs[:, 1]).long().to(F1.device)
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.cpu().numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.cpu().numpy()], hash_seed)

    mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
    neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
    neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
    return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

  def _train_epoch(self, epoch):
    gc.collect()
    self.model.train()
    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    start_iter = (epoch - 1) * len(data_loader)
    
    for curr_iter in range(len(data_loader)):
      input_dict = next(data_loader_iter)
      
      self.optimizer.zero_grad()

      if 'point0' in input_dict and 'point1' in input_dict:
          point0_dict = {
              'coord': input_dict['point0']['coord'].to(self.device),
              'feat': input_dict['point0']['feat'].to(self.device),
              'offset': input_dict['point0']['offset'].to(self.device) if 'offset' in input_dict['point0'] else None,
              'batch_list': input_dict['point0']['batch_list'].to(self.device) if 'batch_list' in input_dict['point0'] else None,
              'grid_size': input_dict['point0']['grid_size'] if 'grid_size' in input_dict['point0'] else self.config.voxel_size,
          }
          point0_dict = {k: v for k, v in point0_dict.items() if v is not None}
          
          point1_dict = {
              'coord': input_dict['point1']['coord'].to(self.device),
              'feat': input_dict['point1']['feat'].to(self.device),
              'offset': input_dict['point1']['offset'].to(self.device) if 'offset' in input_dict['point1'] else None,
              'batch_list': input_dict['point1']['batch_list'].to(self.device) if 'batch_list' in input_dict['point1'] else None,
              'grid_size': input_dict['point1']['grid_size'] if 'grid_size' in input_dict['point1'] else self.config.voxel_size,
          }
          point1_dict = {k: v for k, v in point1_dict.items() if v is not None}
          
          model_output0 = self.model(point0_dict)
          F0 = extract_features(model_output0)
          
          model_output1 = self.model(point1_dict)
          F1 = extract_features(model_output1)
          
      elif 'pcd0' in input_dict and 'pcd1' in input_dict:
          point0_dict = {
              'coord': input_dict['pcd0'].to(self.device),
              'feat': input_dict['sinput0_F'].to(self.device),
              'offset': torch.cumsum(torch.tensor([0] + [len(input_dict['pcd0'])]), dim=0).to(self.device),
              'batch_list': torch.tensor([len(input_dict['pcd0'])]).to(self.device),
              'grid_size': self.config.voxel_size,
          }
          
          point1_dict = {
              'coord': input_dict['pcd1'].to(self.device),
              'feat': input_dict['sinput1_F'].to(self.device),
              'offset': torch.cumsum(torch.tensor([0] + [len(input_dict['pcd1'])]), dim=0).to(self.device),
              'batch_list': torch.tensor([len(input_dict['pcd1'])]).to(self.device),
              'grid_size': self.config.voxel_size,
          }
          
          model_output0 = self.model(point0_dict)
          F0 = extract_features(model_output0)
          
          model_output1 = self.model(point1_dict)
          F1 = extract_features(model_output1)
          
      else:
          if 'sinput0_offset' in input_dict:
              offset0 = input_dict['sinput0_offset'].to(self.device)
          else:
              offset0 = torch.tensor([0, input_dict['sinput0_F'].shape[0]], dtype=torch.long, device=self.device)
          
          if 'sinput1_offset' in input_dict:
              offset1 = input_dict['sinput1_offset'].to(self.device)
          else:
              offset1 = torch.tensor([0, input_dict['sinput1_F'].shape[0]], dtype=torch.long, device=self.device)
          
          sinput0_dict = {
              'feat': input_dict['sinput0_F'].to(self.device),
              'coord': input_dict['sinput0_C'].to(self.device),
              'offset': offset0,
              'grid_size': self.config.voxel_size,
          }
          model_output0 = self.model(sinput0_dict)
          F0 = extract_features(model_output0)

          sinput1_dict = {
              'feat': input_dict['sinput1_F'].to(self.device),
              'coord': input_dict['sinput1_C'].to(self.device),
              'offset': offset1,
              'grid_size': self.config.voxel_size,
          }
          model_output1 = self.model(sinput1_dict)
          F1 = extract_features(model_output1)

      pos_pairs = input_dict['correspondences']
      pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
          F0,
          F1,
          pos_pairs,
          num_pos=self.config.num_pos_per_batch * self.config.batch_size,
          num_hn_samples=self.config.num_hn_samples_per_batch *
          self.config.batch_size)

      loss = pos_loss + self.neg_weight * neg_loss
      loss.backward()
      self.optimizer.step()
      
      gc.collect()
      torch.cuda.empty_cache()

      total_loss += loss.item()
      total_num += 1.0

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', loss.item(), start_iter + curr_iter)
        self.writer.add_scalar('train/pos_loss', pos_loss.item(), start_iter + curr_iter)
        self.writer.add_scalar('train/neg_loss', neg_loss.item(), start_iter + curr_iter)
        log_extra = ""
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}"
            .format(epoch, curr_iter, len(self.data_loader), 
                    loss.item(), pos_loss.item(), neg_loss.item()))

class TripletLossTrainer(ContrastiveLossTrainer):

  def triplet_loss(self,
                   F0,
                   F1,
                   positive_pairs,
                   num_pos=1024,
                   num_hn_samples=None,
                   num_rand_triplet=1024):
    """
    Generate negative pairs
    """
    N0, N1 = F0.shape[0] if hasattr(F0, 'shape') else len(F0), F1.shape[0] if hasattr(F1, 'shape') else len(F1)
    num_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)

    if num_pos_pairs > num_pos:
      pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs
    
    if not isinstance(sample_pos_pairs, np.ndarray):
      sample_pos_pairs = np.array(sample_pos_pairs, dtype=np.int64)

    pos_ind0 = torch.from_numpy(sample_pos_pairs[:, 0]).long().to(F0.device)
    pos_ind1 = torch.from_numpy(sample_pos_pairs[:, 1]).long().to(F1.device)
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)
    pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

    # Random triplets
    rand_inds = np.random.choice(
        num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)
    rand_pairs = positive_pairs[rand_inds]
    negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

    # Remove positives from negatives
    rand_neg_keys = _hash([rand_pairs[:, 0], negatives], hash_seed)
    rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
    anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
    negatives = negatives[rand_mask]

    rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
    rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)

    loss = F.relu(rand_pos_dist + self.neg_thresh - rand_neg_dist).mean()

    return loss, pos_dist.mean(), rand_neg_dist.mean()

  def _train_epoch(self, epoch):
    config = self.config

    gc.collect()
    self.model.train()

    # Epoch starts from 1
    total_loss = 0
    total_num = 0.0
    data_loader = self.data_loader
    data_loader_iter = self.data_loader.__iter__()
    iter_size = self.iter_size
    pos_dist_meter, neg_dist_meter = AverageMeter(), AverageMeter()
    start_iter = (epoch - 1) * (len(data_loader) // iter_size)
    for curr_iter in range(len(data_loader) // iter_size):
      self.optimizer.zero_grad()
      batch_loss = 0
      for iter_idx in range(iter_size):
        input_dict = next(data_loader_iter)

        # pairs consist of (xyz1 index, xyz0 index)
        # Convert to pointcnnpp format
        sinput0_dict = {
            'feat': input_dict['sinput0_F'].to(self.device),
            'coord': input_dict['sinput0_C'].to(self.device),
            'offset': input_dict.get('sinput0_offset', torch.tensor([0, input_dict['sinput0_F'].shape[0]], dtype=torch.long, device=self.device)),
            'grid_size': self.config.voxel_size,
        }
        model_output0 = self.model(sinput0_dict)
        F0 = extract_features(model_output0)

        sinput1_dict = {
            'feat': input_dict['sinput1_F'].to(self.device),
            'coord': input_dict['sinput1_C'].to(self.device),
            'offset': input_dict.get('sinput1_offset', torch.tensor([0, input_dict['sinput1_F'].shape[0]], dtype=torch.long, device=self.device)),
            'grid_size': self.config.voxel_size,
        }
        model_output1 = self.model(sinput1_dict)
        F1 = extract_features(model_output1)

        pos_pairs = input_dict['correspondences']
        loss, pos_dist, neg_dist = self.triplet_loss(
            F0,
            F1,
            pos_pairs,
            num_pos=config.triplet_num_pos * config.batch_size,
            num_hn_samples=config.triplet_num_hn * config.batch_size,
            num_rand_triplet=config.triplet_num_rand * config.batch_size)
        loss /= iter_size
        loss.backward()
        batch_loss += loss.item()
        pos_dist_meter.update(pos_dist)
        neg_dist_meter.update(neg_dist)

      self.optimizer.step()
      gc.collect()

      torch.cuda.empty_cache()

      total_loss += batch_loss
      total_num += 1.0

      if curr_iter % self.config.stat_freq == 0:
        self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
        logging.info(
            "Train Epoch: {} [{}/{}], Current Loss: {:.3e}, Pos dist: {:.3e}, Neg dist: {:.3e}"
            .format(epoch, curr_iter,
                    len(self.data_loader) //
                    iter_size, batch_loss, pos_dist_meter.avg, neg_dist_meter.avg))


class HardestTripletLossTrainer(TripletLossTrainer):

  def triplet_loss(self,
                   F0,
                   F1,
                   positive_pairs,
                   num_pos=1024,
                   num_hn_samples=512,
                   num_rand_triplet=1024):
    """
    Generate negative pairs
    """
    N0, N1 = F0.shape[0] if hasattr(F0, 'shape') else len(F0), F1.shape[0] if hasattr(F1, 'shape') else len(F1)
    num_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if num_pos_pairs > num_pos:
      pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
      sample_pos_pairs = positive_pairs[pos_sel]
    else:
      sample_pos_pairs = positive_pairs
    
    if not isinstance(sample_pos_pairs, np.ndarray):
      sample_pos_pairs = np.array(sample_pos_pairs, dtype=np.int64)

    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = torch.from_numpy(sample_pos_pairs[:, 0]).long().to(F0.device)
    pos_ind1 = torch.from_numpy(sample_pos_pairs[:, 1]).long().to(F1.device)
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
      positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.cpu().numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.cpu().numpy()], hash_seed)

    mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

    # Random triplets
    rand_inds = np.random.choice(
        num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)
    rand_pairs = positive_pairs[rand_inds]
    negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

    # Remove positives from negatives
    rand_neg_keys = _hash([rand_pairs[:, 0], negatives], hash_seed)
    rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
    anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
    negatives = negatives[rand_mask]

    rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
    rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)

    loss = F.relu(
        torch.cat([
            rand_pos_dist + self.neg_thresh - rand_neg_dist,
            pos_dist[mask0] + self.neg_thresh - D01min[mask0],
            pos_dist[mask1] + self.neg_thresh - D10min[mask1]
        ])).mean()

    return loss, pos_dist.mean(), (D01min.mean() + D10min.mean()).item() / 2
