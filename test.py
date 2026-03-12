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
from util.pointcloud import make_open3d_point_cloud, make_open3d_feature
from lib.timer import AverageMeter, Timer

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
  success_meter, rte_meter, rre_meter = AverageMeter(), AverageMeter(), AverageMeter()
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
    xyz0, xyz1 = data_dict['pcd0'], data_dict['pcd1']
    T_gth = data_dict['T_gt']
    xyz0np, xyz1np = xyz0.numpy(), xyz1.numpy()
    pcd0 = make_open3d_point_cloud(xyz0np)
    pcd1 = make_open3d_point_cloud(xyz1np)
    with torch.no_grad():
      feat_timer.tic()
      sinput0_dict = {
          'feat': data_dict['sinput0_F'].to(device),
          'coord': data_dict['sinput0_C'].to(device),
          'offset': data_dict.get('sinput0_offset', torch.tensor([0, data_dict['sinput0_F'].shape[0]], dtype=torch.long, device=device)),
          'grid_size': config.voxel_size,
      }
      model_output0 = model(sinput0_dict)
      F0 = extract_features(model_output0).detach()
      sinput1_dict = {
          'feat': data_dict['sinput1_F'].to(device),
          'coord': data_dict['sinput1_C'].to(device),
          'offset': data_dict.get('sinput1_offset', torch.tensor([0, data_dict['sinput1_F'].shape[0]], dtype=torch.long, device=device)),
          'grid_size': config.voxel_size,
      }
      model_output1 = model(sinput1_dict)
      F1 = extract_features(model_output1).detach()
      feat_timer.toc()
    
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
    rte = np.linalg.norm(T_ransac[:3, 3] - T_gth[:3, 3])
    rre = np.arccos((np.trace(T_ransac[:3, :3].t() @ T_gth[:3, :3]) - 1) / 2)
    if rte < 2:
      rte_meter.update(rte)
    if not np.isnan(rre) and rre < np.pi / 180 * 5:
      rre_meter.update(rre)
    if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
      success_meter.update(1)
    else:
      success_meter.update(0)
      logging.info(f"Failed with RTE: {rte}, RRE: {rre}")
    if i % 10 == 0:
      logging.info(
          f"{i} / {N}: Data time: {data_timer.avg}, Feat time: {feat_timer.avg}," +
          f" Reg time: {reg_timer.avg}, RTE: {rte_meter.avg}," +
          f" RRE: {rre_meter.avg}, Success: {success_meter.sum} / {success_meter.count}"
          + f" ({success_meter.avg * 100} %)")
      data_timer.reset()
      feat_timer.reset()
      reg_timer.reset()
  logging.info(
      f"RTE: {rte_meter.avg}, var: {rte_meter.var}," +
      f" RRE: {rre_meter.avg}, var: {rre_meter.var}, Success: {success_meter.sum} " +
      f"/ {success_meter.count} ({success_meter.avg * 100} %)")
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
    if 'threed_match_dir' not in config or config.threed_match_dir is None:
      raise ValueError("threed_match_dir must be specified either in config.json or via --threed_match_dir argument for 3DMatch dataset")
    logging.info(f"Testing on 3DMatch dataset: {dataset_name}")
    logging.info(f"3DMatch root: {config.threed_match_dir}")
  else:
    logging.warning(f"Unknown dataset type: {dataset_name}, proceeding with config from checkpoint")
  
  main(config)
