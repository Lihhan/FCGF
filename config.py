import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


logging_arg = add_argument_group('Logging')
logging_arg.add_argument('--out_dir', type=str, default='outputs')

trainer_arg = add_argument_group('Trainer')
trainer_arg.add_argument('--trainer', type=str, default='HardestContrastiveLossTrainer')
trainer_arg.add_argument('--save_freq_epoch', type=int, default=1)
trainer_arg.add_argument('--batch_size', type=int, default=1)
trainer_arg.add_argument('--val_batch_size', type=int, default=1)

# Hard negative mining
trainer_arg.add_argument('--use_hard_negative', type=str2bool, default=True)
trainer_arg.add_argument('--hard_negative_sample_ratio', type=int, default=0.05)
trainer_arg.add_argument('--hard_negative_max_num', type=int, default=3000)
trainer_arg.add_argument('--num_pos_per_batch', type=int, default=1024)
trainer_arg.add_argument('--num_hn_samples_per_batch', type=int, default=256)

# Metric learning loss
trainer_arg.add_argument('--neg_thresh', type=float, default=1.4)
trainer_arg.add_argument('--pos_thresh', type=float, default=0.1)
trainer_arg.add_argument('--neg_weight', type=float, default=1)

# Data augmentation
trainer_arg.add_argument('--use_random_scale', type=str2bool, default=False)
trainer_arg.add_argument('--min_scale', type=float, default=0.8)
trainer_arg.add_argument('--max_scale', type=float, default=1.2)
trainer_arg.add_argument('--use_random_rotation', type=str2bool, default=True)
trainer_arg.add_argument('--rotation_range', type=float, default=360)

# Data loader configs
trainer_arg.add_argument('--train_phase', type=str, default="train")
trainer_arg.add_argument('--val_phase', type=str, default="val")
trainer_arg.add_argument('--test_phase', type=str, default="test")

trainer_arg.add_argument('--stat_freq', type=int, default=40)
trainer_arg.add_argument('--profile_batch_time', type=str2bool, default=True,
                         help='Record forward/backward time per batch during training')
trainer_arg.add_argument('--test_valid', type=str2bool, default=True)
trainer_arg.add_argument('--val_max_iter', type=int, default=400)
trainer_arg.add_argument('--val_epoch_freq', type=int, default=1)
trainer_arg.add_argument(
    '--positive_pair_search_voxel_size_multiplier', type=float, default=1.5)

trainer_arg.add_argument('--hit_ratio_thresh', type=float, default=0.1)

# Triplets
trainer_arg.add_argument('--triplet_num_pos', type=int, default=256)
trainer_arg.add_argument('--triplet_num_hn', type=int, default=512)
trainer_arg.add_argument('--triplet_num_rand', type=int, default=1024)

# dNetwork specific configurations
net_arg = add_argument_group('Network')
net_arg.add_argument('--model', type=str, default='ResUNetBN2C')
net_arg.add_argument('--model_n_out', type=int, default=32, help='Feature dimension')
net_arg.add_argument('--conv1_kernel_size', type=int, default=5)
net_arg.add_argument('--normalize_feature', type=str2bool, default=True)
net_arg.add_argument('--dist_type', type=str, default='L2')
net_arg.add_argument('--best_val_metric', type=str, default='feat_match_ratio')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--max_epoch', type=int, default=100)
opt_arg.add_argument('--lr', type=float, default=1e-1)
opt_arg.add_argument('--momentum', type=float, default=0.8)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.05)
opt_arg.add_argument('--exp_gamma', type=float, default=0.99)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
opt_arg.add_argument(
    '--icp_cache_path', type=str, default="/home/chrischoy/datasets/FCGF/kitti/icp/")

misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--weights', type=str, default=None)
misc_arg.add_argument('--weights_dir', type=str, default=None)
misc_arg.add_argument('--resume', type=str, default=None)
misc_arg.add_argument('--resume_dir', type=str, default=None)
misc_arg.add_argument('--train_num_thread', type=int, default=6)
misc_arg.add_argument('--val_num_thread', type=int, default=1)
misc_arg.add_argument('--test_num_thread', type=int, default=2)
misc_arg.add_argument('--fast_validation', type=str2bool, default=False)
misc_arg.add_argument(
    '--nn_max_n',
    type=int,
    default=500,
    help='The maximum number of features to find nearest neighbors in batch')

# Dataset specific configurations
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='ThreeDMatchNewPairDatasetPure')
# KITTI: 0.3, 3DMatch: 0.025
data_arg.add_argument('--voxel_size', type=float, default=None)
data_arg.add_argument('--min_dist', type=float, default=4)
data_arg.add_argument(
    '--threed_match_dir', type=str, default=None,
    help='Root directory for 3DMatch dataset. Defaults to THREEDMATCH_ROOT env var.')
data_arg.add_argument(
    '--threedmatch_root', type=str, default=None,
    help='Root directory for 3DMatch dataset (OverlapPredator style). Defaults to THREEDMATCH_ROOT env var.')
data_arg.add_argument('--train_info', type=str, default=None,
                      help='Path to train_info.pkl file (OverlapPredator format). Defaults to FCGF/configs/indoor/train_info.pkl')
data_arg.add_argument('--val_info', type=str, default=None,
                      help='Path to val_info.pkl file (OverlapPredator format). Defaults to FCGF/configs/indoor/val_info.pkl')
data_arg.add_argument('--overlap_radius', type=float, default=0.0375,
                      help='Overlap radius for positive pair search')
data_arg.add_argument('--augment_noise', type=float, default=0.005,
                      help='Gaussian noise augmentation strength')
data_arg.add_argument('--max_points', type=int, default=1000000,
                      help='Maximum number of points per point cloud')
data_arg.add_argument('--rot_factor', type=float, default=1.0,
                      help='Rotation factor for data augmentation')
data_arg.add_argument('--pre_downsample_voxel_size', type=float, default=0.02)
data_arg.add_argument(
    '--kitti_root', type=str, default="/home/chrischoy/datasets/FCGF/kitti/")
data_arg.add_argument(
    '--kitti_max_time_diff',
    type=int,
    default=3,
    help='max time difference between pairs (non inclusive)')
data_arg.add_argument('--kitti_date', type=str, default='2011_09_26')


def get_config():
  import os
  args = parser.parse_args()
  if args.dataset == 'KITTINMPairDataset':
    args.voxel_size = 0.3
  elif args.dataset == 'ThreeDMatchNewPairDatasetPure':
    args.voxel_size = 0.025
  else:
    raise ValueError(f"Unsupported dataset: {args.dataset}")
  # Resolve 3DMatch root from THREEDMATCH_ROOT env when not set via args
  threed_root = os.environ.get('THREEDMATCH_ROOT', '')
  if args.threedmatch_root is None or args.threedmatch_root == '':
    args.threedmatch_root = threed_root if threed_root else None
  if args.threed_match_dir is None or args.threed_match_dir == '':
    args.threed_match_dir = threed_root if threed_root else None
  # Resolve train_info/val_info defaults (relative to FCGF module)
  _fcgf_dir = os.path.dirname(os.path.abspath(__file__))
  if args.train_info is None or args.train_info == '':
    args.train_info = os.path.join(_fcgf_dir, 'configs', 'indoor', 'train_info.pkl')
  if args.val_info is None or args.val_info == '':
    args.val_info = os.path.join(_fcgf_dir, 'configs', 'indoor', 'val_info.pkl')
  return args
