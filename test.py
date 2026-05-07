import open3d as o3d
import sys
import logging
import json
import argparse
import numpy as np
from easydict import EasyDict as edict
import torch
from model import load_model
from lib.data_loaders import make_data_loader
from lib.eval import find_nn_gpu
from util.pointcloud import make_open3d_point_cloud, make_open3d_feature
from lib.timer import AverageMeter, Timer


def extract_coords(model_output, fallback_coords=None):
    """Extract coordinates from model output. Identical to trainer.extract_coords."""
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

def extract_features(model_output):
    """Extract features from model output (supports both dict and tensor formats)"""
    if isinstance(model_output, dict):
        if 'feat' in model_output:
            return model_output['feat']
        elif 'F' in model_output:
            return model_output['F']
    if hasattr(model_output, 'F'):
        return model_output.F
    if hasattr(model_output, 'feat'):
        return model_output.feat
    if hasattr(model_output, 'shape'):
        return model_output
    raise ValueError(f"Cannot extract features from type: {type(model_output)}")
ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

def apply_transform(pts, trans):
  """Apply a 4x4 transformation to points. Identical to trainer.apply_transform."""
  R = trans[:3, :3]
  T = trans[:3, 3]
  if isinstance(R, torch.Tensor):
    R_T = R.t()
  else:
    R_T = R.T
  return pts @ R_T + T


def find_corr(xyz0, xyz1, F0, F1, nn_max_n=500, subsample_size=1000, mutual=True):
  """Find correspondences via nearest neighbor in feature space.
  If mutual=True, only keep mutual nearest neighbors (A->B and B->A).
  Then subsample from the mutual matches."""
  # F0 -> F1: for each point in F0, find NN in F1
  nn_result_01 = find_nn_gpu(F0, F1, nn_max_n=nn_max_n, return_distance=True, dist_type='SquareL2')
  if isinstance(nn_result_01, tuple):
    nn_inds_01, nn_dists_01 = nn_result_01
  else:
    nn_inds_01 = nn_result_01
    nn_dists_01 = None

  if isinstance(nn_inds_01, torch.Tensor):
    nn_inds_01 = nn_inds_01.numpy()
  if nn_dists_01 is not None and isinstance(nn_dists_01, torch.Tensor):
    nn_dists_01 = nn_dists_01.numpy()

  N0 = F0.shape[0] if hasattr(F0, 'shape') else len(F0)
  N1 = F1.shape[0] if hasattr(F1, 'shape') else len(F1)

  if mutual:
    # F1 -> F0: for each point in F1, find NN in F0
    nn_result_10 = find_nn_gpu(F1, F0, nn_max_n=nn_max_n, return_distance=False, dist_type='SquareL2')
    if isinstance(nn_result_10, tuple):
      nn_inds_10 = nn_result_10[0]
    else:
      nn_inds_10 = nn_result_10
    if isinstance(nn_inds_10, torch.Tensor):
      nn_inds_10 = nn_inds_10.numpy()

    # Mutual check (vectorized): i->j and j->i
    valid_01 = (nn_inds_01 >= 0) & (nn_inds_01 < N1)
    # For each valid i, check if nn_inds_10[nn_inds_01[i]] == i
    mutual_mask = np.zeros(N0, dtype=bool)
    valid_idx = np.where(valid_01)[0]
    if len(valid_idx) > 0:
      j_vals = nn_inds_01[valid_idx]
      back_map = nn_inds_10[j_vals]
      mutual_mask[valid_idx] = (back_map == valid_idx)
    valid_mask = mutual_mask
  else:
    valid_mask = (nn_inds_01 >= 0) & (nn_inds_01 < N1)

  if valid_mask.sum() == 0:
    logging.warning(f"find_corr: no valid matches found (N0={N0}, N1={N1}, mutual={mutual})")
    if isinstance(xyz0, torch.Tensor):
      xyz0 = xyz0.cpu().numpy()
    if isinstance(xyz1, torch.Tensor):
      xyz1 = xyz1.cpu().numpy()
    return np.empty((0, 3), dtype=xyz0.dtype), np.empty((0, 3), dtype=xyz1.dtype)

  nn_inds = nn_inds_01[valid_mask]
  valid_inds0 = np.where(valid_mask)[0]

  if subsample_size > 0 and len(valid_inds0) > subsample_size:
    if nn_dists_01 is not None:
      valid_dists = nn_dists_01[valid_mask]
      if isinstance(valid_dists, torch.Tensor):
        valid_dists = valid_dists.squeeze().cpu().numpy()
      else:
        valid_dists = np.asarray(valid_dists).squeeze()
      if valid_dists.ndim > 1:
        valid_dists = valid_dists.flatten()
      # Top-k: select the ones with smallest feature distance
      topk_idx = np.argsort(valid_dists)[:subsample_size]
      valid_inds0 = valid_inds0[topk_idx]
      nn_inds = nn_inds[topk_idx]
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


def evaluate_hit_ratio(xyz0, xyz1, T_gth, thresh=0.1):
  """Compute hit ratio. Identical to trainer.evaluate_hit_ratio."""
  if isinstance(xyz0, torch.Tensor):
    xyz0 = xyz0.cpu().numpy()
  if isinstance(xyz1, torch.Tensor):
    xyz1 = xyz1.cpu().numpy()
  if isinstance(T_gth, torch.Tensor):
    T_gth = T_gth.cpu().numpy()

  xyz0 = apply_transform(xyz0, T_gth)
  dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
  return (dist < thresh).astype(np.float32).mean()


def compute_registration_rmse(xyz0, T_est, T_gt):
  """Compute RMSE between T_est and T_gt applied to xyz0.
  RMSE = sqrt(mean(||T_est @ p - T_gt @ p||^2))
  Standard 3DMatch Registration Recall criterion: RMSE < 0.2m.
  """
  if isinstance(xyz0, torch.Tensor):
    xyz0 = xyz0.cpu().numpy()
  if isinstance(T_est, torch.Tensor):
    T_est = T_est.cpu().numpy()
  if isinstance(T_gt, torch.Tensor):
    T_gt = T_gt.cpu().numpy()
  xyz0_est = (T_est[:3, :3] @ xyz0.T).T + T_est[:3, 3]
  xyz0_gt = (T_gt[:3, :3] @ xyz0.T).T + T_gt[:3, 3]
  rmse = np.sqrt(((xyz0_est - xyz0_gt)**2).sum(1).mean())
  return rmse


def main(config):
  test_loader = make_data_loader(
      config, config.test_phase, 1, num_threads=config.test_num_thread, shuffle=True)
  num_feats = 1
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  Model = load_model(config.model)
  model = Model(
      num_feats,
      config.model_n_out,
      bn_momentum=config.bn_momentum,
      conv1_kernel_size=config.conv1_kernel_size,
      normalize_feature=config.normalize_feature)
  checkpoint = torch.load(config.save_dir + '/checkpoint.pth', weights_only=False)
  model.load_state_dict(checkpoint['state_dict'])
  model = model.to(device)
  model.eval()
  np.random.seed(0)
  hit_ratio_thresh = getattr(config, 'hit_ratio_thresh', 0.1)
  success_meter, rte_meter, rre_meter = AverageMeter(), AverageMeter(), AverageMeter()
  hit_ratio_meter, fmr_meter = AverageMeter(), AverageMeter()
  data_timer, feat_timer, reg_timer = Timer(), Timer(), Timer()
  test_iter = iter(test_loader)
  N = len(test_loader)
  n_gpu_failures = 0
  for i in range(N):
    data_timer.tic()
    try:
      data_dict = next(test_iter)
    except ValueError:
      n_gpu_failures += 1
      logging.info(f"# Erroneous GPU Pair {n_gpu_failures}")
      continue
    data_timer.toc()
    T_gth = data_dict['T_gt']
    with torch.no_grad():
      feat_timer.tic()
      # Use pcd0 as coord (identical to validation branch 1 in trainer._valid_epoch)
      offset0 = torch.cumsum(torch.tensor([0] + [len(data_dict['pcd0'])], dtype=torch.long), dim=0).to(device)
      offset1 = torch.cumsum(torch.tensor([0] + [len(data_dict['pcd1'])], dtype=torch.long), dim=0).to(device)

      point0_dict = {
          'coord': data_dict['pcd0'].to(device),
          'feat': data_dict['sinput0_F'].to(device),
          'offset': offset0,
          'grid_size': config.voxel_size,
      }
      model_output0 = model(point0_dict)
      F0 = extract_features(model_output0).detach()
      xyz0_model = extract_coords(model_output0, data_dict['pcd0'])
      if xyz0_model is None:
          xyz0_model = data_dict['pcd0']

      point1_dict = {
          'coord': data_dict['pcd1'].to(device),
          'feat': data_dict['sinput1_F'].to(device),
          'offset': offset1,
          'grid_size': config.voxel_size,
      }
      model_output1 = model(point1_dict)
      F1 = extract_features(model_output1).detach()
      xyz1_model = extract_coords(model_output1, data_dict['pcd1'])
      if xyz1_model is None:
          xyz1_model = data_dict['pcd1']
      feat_timer.toc()

    # Get coordinates aligned with features (identical to validation)
    if isinstance(xyz0_model, torch.Tensor):
        xyz0 = xyz0_model.cpu().numpy()
    else:
        xyz0 = np.array(xyz0_model, dtype=np.float32)
    if isinstance(xyz1_model, torch.Tensor):
        xyz1 = xyz1_model.cpu().numpy()
    else:
        xyz1 = np.array(xyz1_model, dtype=np.float32)

    xyz0_tensor = data_dict['pcd0']
    xyz1_tensor = data_dict['pcd1']
    xyz0np = xyz0_tensor.numpy() if isinstance(xyz0_tensor, torch.Tensor) else xyz0
    xyz1np = xyz1_tensor.numpy() if isinstance(xyz1_tensor, torch.Tensor) else xyz1
    pcd0 = make_open3d_point_cloud(xyz0np)
    pcd1 = make_open3d_point_cloud(xyz1np)
    
    # Ensure features are on CPU for Open3D
    if isinstance(F0, torch.Tensor):
        F0 = F0.cpu()
    if isinstance(F1, torch.Tensor):
        F1 = F1.cpu()
    
    feat0 = make_open3d_feature(F0.numpy() if isinstance(F0, torch.Tensor) else F0, config.model_n_out, F0.shape[0])
    feat1 = make_open3d_feature(F1.numpy() if isinstance(F1, torch.Tensor) else F1, config.model_n_out, F1.shape[0])
    reg_timer.tic()
    distance_threshold = config.voxel_size * 1.0
    try:
      # New Open3D API: mutual_filter parameter added after features
      ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
          pcd0, pcd1, feat0, feat1, 
          mutual_filter=False,  # Add mutual_filter parameter
          max_correspondence_distance=distance_threshold,
          estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
          ransac_n=4,
          checkers=[
              o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
              o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
          ],
          criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 10000))
    except (AttributeError, TypeError):
      # Fallback to old API or handle other errors
      try:
        ransac_result = o3d.registration.registration_ransac_based_on_feature_matching(
            pcd0, pcd1, feat0, feat1, distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.registration.RANSACConvergenceCriteria(4000000, 10000))
      except (AttributeError, TypeError):
        # Try with new API but positional arguments
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd0, pcd1, feat0, feat1, False, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 10000))
    T_ransac = torch.from_numpy(ransac_result.transformation.astype(np.float32))
    reg_timer.toc()

    # Compute hit ratio and feature match ratio (identical to trainer._valid_epoch)
    xyz0_corr, xyz1_corr = find_corr(
        xyz0, xyz1, F0.to(device), F1.to(device),
        nn_max_n=getattr(config, 'nn_max_n', 500),
        subsample_size=1000)
    if len(xyz0_corr) > 0:
      if isinstance(xyz0_corr, np.ndarray):
        xyz0_corr = torch.from_numpy(xyz0_corr).float()
      if isinstance(xyz1_corr, np.ndarray):
        xyz1_corr = torch.from_numpy(xyz1_corr).float()
      hit_ratio = evaluate_hit_ratio(xyz0_corr, xyz1_corr, T_gth, thresh=hit_ratio_thresh)
      hit_ratio_meter.update(hit_ratio)
      fmr_meter.update(hit_ratio > 0.05)

    rte = np.linalg.norm(T_ransac[:3, 3] - T_gth[:3, 3])
    rre = np.arccos(np.clip((np.trace(T_ransac[:3, :3].t() @ T_gth[:3, :3]) - 1) / 2, -1, 1))
    rte_meter.update(rte)
    if not np.isnan(rre):
      rre_meter.update(rre)

    # Registration Recall: RMSE < 0.2m (standard 3DMatch benchmark criterion)
    rmse = compute_registration_rmse(xyz0, T_ransac, T_gth)
    if rmse < 0.2:
      success_meter.update(1)
    else:
      success_meter.update(0)
      logging.info(f"Failed with RMSE: {rmse:.4f}, RTE: {rte:.4f}, RRE: {np.degrees(rre):.2f}deg")
    if i % 10 == 0:
      logging.info(
          f"{i} / {N}: Data time: {data_timer.avg}, Feat time: {feat_timer.avg}," +
          f" Reg time: {reg_timer.avg}, RTE: {rte_meter.avg}," +
          f" RRE: {rre_meter.avg}, Success: {success_meter.sum} / {success_meter.count}" +
          f" ({success_meter.avg * 100} %)," +
          f" Hit Ratio: {hit_ratio_meter.avg:.4f}, FMR: {fmr_meter.avg:.4f}")
      data_timer.reset()
      feat_timer.reset()
      reg_timer.reset()
  logging.info(
      f"RTE: {rte_meter.avg}, var: {rte_meter.var}," +
      f" RRE: {rre_meter.avg}, var: {rre_meter.var}, Success: {success_meter.sum} " +
      f"/ {success_meter.count} ({success_meter.avg * 100} %)," +
      f" Hit Ratio: {hit_ratio_meter.avg:.4f}, FMR: {fmr_meter.avg:.4f}")
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Test FCGF model on KITTI or 3DMatch dataset')
  parser.add_argument('--save_dir', default=None, type=str, required=True,
                      help='Directory containing checkpoint and config.json')
  parser.add_argument('--test_phase', default='test', type=str,
                      help='Test phase: test, val, etc.')
  parser.add_argument('--test_num_thread', default=5, type=int,
                      help='Number of threads for data loading')
  parser.add_argument('--kitti_root', type=str, default=None,
                      help='KITTI dataset root directory (only needed for KITTI dataset)')
  parser.add_argument('--threed_match_dir', type=str, default=None,
                      help='3DMatch dataset root directory (only needed for 3DMatch dataset)')
  parser.add_argument('--test_info', type=str, default=None,
                      help='Path to test info pkl file (e.g. configs/indoor/3DMatch.pkl or 3DLoMatch.pkl)')
  parser.add_argument('--dataset', type=str, default=None,
                      help='Override dataset class name (e.g. ThreeDMatchNewPairDatasetPure)')
  args = parser.parse_args()
  
  # Load config from checkpoint directory
  config = json.load(open(args.save_dir + '/config.json', 'r'))
  config = edict(config)
  config.save_dir = args.save_dir
  config.test_phase = args.test_phase
  config.test_num_thread = args.test_num_thread
  
  # Override dataset type if specified
  if args.dataset is not None:
    config.dataset = args.dataset
  
  # Override test info pkl if specified (for 3DMatch/3DLoMatch benchmark)
  if args.test_info is not None:
    config.train_info = args.test_info
    config.val_info = args.test_info
    logging.info(f"Using test info pkl: {args.test_info}")
  
  # Ensure dataset is specified
  if 'dataset' not in config or config.dataset is None:
    raise ValueError("dataset must be specified either in config.json or via --dataset argument")
  
  # Set dataset-specific configurations
  dataset_name = config.dataset
  
  if 'KITTI' in dataset_name:
    # KITTI dataset configuration
    if args.kitti_root is not None:
      config.kitti_root = args.kitti_root
    if 'kitti_root' not in config or config.kitti_root is None:
      raise ValueError("kitti_root must be specified either in config.json or via --kitti_root argument for KITTI dataset")
    config.kitti_odometry_root = config.kitti_root + '/dataset'
    logging.info(f"Testing on KITTI dataset: {dataset_name}")
    logging.info(f"KITTI root: {config.kitti_root}")
  elif 'ThreeDMatch' in dataset_name or '3DMatch' in dataset_name:
    # 3DMatch dataset configuration
    if args.threed_match_dir is not None:
      config.threed_match_dir = args.threed_match_dir
      config.threedmatch_root = args.threed_match_dir
    if 'threed_match_dir' not in config or config.threed_match_dir is None:
      raise ValueError("threed_match_dir must be specified either in config.json or via --threed_match_dir argument for 3DMatch dataset")
    config.threedmatch_root = config.threed_match_dir
    logging.info(f"Testing on 3DMatch dataset: {dataset_name}")
    logging.info(f"3DMatch root: {config.threed_match_dir}")
  else:
    logging.warning(f"Unknown dataset type: {dataset_name}, proceeding with config from checkpoint")
  
  main(config)
