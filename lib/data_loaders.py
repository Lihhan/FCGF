import logging
import random
import torch
import torch.utils.data
import numpy as np
import glob
import os
from scipy.linalg import expm, norm
import pathlib
import pickle
import copy

from util.pointcloud import get_matching_indices, make_open3d_point_cloud
from util.trajectory import read_trajectory
import lib.transforms as t

import open3d as o3d

try:
    from pointcnnpp.internals.grid_sample import grid_sample_filter
except ImportError:
    grid_sample_filter = None
    logging.warning("pointcnnpp.internals.grid_sample not available, voxel_downsample_using_pointcloud_downsampler will fail")

kitti_cache = {}
kitti_icp_cache = {}
threedmatch_icp_cache = {}


def _vectorized_matching_indices(matching_inds_list, batch_list0, batch_list1):
    cumsum0 = torch.cumsum(torch.cat([torch.tensor([0], device=batch_list0.device), batch_list0[:-1]]), dim=0)
    cumsum1 = torch.cumsum(torch.cat([torch.tensor([0], device=batch_list1.device), batch_list1[:-1]]), dim=0)
    
    matching_tensors = []
    for i, matching_inds in enumerate(matching_inds_list):
        if len(matching_inds) > 0:
            if isinstance(matching_inds, np.ndarray):
                matching_arr = matching_inds
            else:
                matching_arr = np.array(matching_inds, dtype=np.int64)
            offset = torch.tensor([[cumsum0[i], cumsum1[i]]], device=batch_list0.device)
            matching_tensor = torch.from_numpy(matching_arr) + offset
            matching_tensors.append(matching_tensor)
    
    if matching_tensors:
        return torch.cat(matching_tensors, dim=0).int()
    else:
        return torch.empty((0, 2), dtype=torch.int, device=batch_list0.device)


def make_collate_pair_fn(voxel_size=0.3):
  def collate_pair_fn(list_data):
    first_item = list_data[0]
    has_pair_info = len(first_item) == 9
    
    if has_pair_info:
      xyz0, xyz1, feats0, feats1, matching_inds, trans, _, _, pair_infos = list(zip(*list_data))
    elif len(first_item) == 8 and isinstance(first_item[7], torch.Tensor):
      xyz0, xyz1, feats0, feats1, matching_inds, trans, _, _ = list(zip(*list_data))
      pair_infos = [("unknown", "unknown")] * len(list_data)
    else:
      xyz0, xyz1, _, _, feats0, feats1, matching_inds, trans = list(zip(*list_data))
      pair_infos = [("unknown", "unknown")] * len(list_data)

    def batch_to_tensor(data_list):
        tensors = []
        for data in data_list:
            if isinstance(data, torch.Tensor):
                tensors.append(data.float())
            elif isinstance(data, np.ndarray):
                tensors.append(torch.from_numpy(data).float())
            else:
                raise ValueError(f'Cannot convert to torch tensor: {type(data)}')
        return tensors
    
    xyz0_tensors = batch_to_tensor(xyz0)
    xyz1_tensors = batch_to_tensor(xyz1)
    feats0_tensors = batch_to_tensor(feats0)
    feats1_tensors = batch_to_tensor(feats1)
    trans_tensors = batch_to_tensor(trans)
    
    batch_list0 = torch.tensor([len(xyz) for xyz in xyz0_tensors], dtype=torch.long)
    batch_list1 = torch.tensor([len(xyz) for xyz in xyz1_tensors], dtype=torch.long)
    
    matching_inds_batch = _vectorized_matching_indices(matching_inds, batch_list0, batch_list1)
    
    len_batch = list(zip(batch_list0.tolist(), batch_list1.tolist()))
    
    xyz_batch0 = torch.cat(xyz0_tensors, dim=0)
    xyz_batch1 = torch.cat(xyz1_tensors, dim=0)
    feat_batch0 = torch.cat(feats0_tensors, dim=0)
    feat_batch1 = torch.cat(feats1_tensors, dim=0)
    trans_batch = torch.cat(trans_tensors, dim=0)
    
    offset0 = torch.cat([torch.tensor([0], device=batch_list0.device), torch.cumsum(batch_list0, dim=0)]).long()
    offset1 = torch.cat([torch.tensor([0], device=batch_list1.device), torch.cumsum(batch_list1, dim=0)]).long()
    
    coords_batch0 = xyz_batch0
    coords_batch1 = xyz_batch1
    feats_batch0 = feat_batch0
    feats_batch1 = feat_batch1

    out = {
        'pcd0': coords_batch0,
        'pcd1': coords_batch1,
        'pcd0_original': xyz_batch0,
        'pcd1_original': xyz_batch1,
        'sinput0_C': coords_batch0,
        'sinput0_F': feat_batch0,
        'sinput0_offset': offset0,
        'sinput1_C': coords_batch1,
        'sinput1_F': feat_batch1,
        'sinput1_offset': offset1,
        'correspondences': matching_inds_batch,
        'T_gt': trans_batch,
        'len_batch': len_batch
    }
    if has_pair_info:
      out['pair_info'] = list(pair_infos)
    return out
  return collate_pair_fn


# Rotation matrix along axis with angle theta
def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
  T = np.eye(4)
  R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
  T[:3, :3] = R
  T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
  return T


def voxel_downsample_using_pointcloud_downsampler(xyz, voxel_size):
    if grid_sample_filter is None:
        raise ImportError("pointcnnpp.internals.grid_sample not available")
    
    if voxel_size is None:
        if isinstance(xyz, torch.Tensor):
            selected_indices = np.arange(len(xyz))
        else:
            selected_indices = np.arange(len(xyz))
        return selected_indices, np.array([len(selected_indices)], dtype=np.int64)
    
    if not isinstance(xyz, torch.Tensor):
        xyz = torch.from_numpy(xyz).float()
    
    device = xyz.device
    
    sample_inds = torch.zeros(len(xyz), dtype=torch.long, device=device)
    
    _, _, indices, _ = grid_sample_filter(
        points=xyz,
        grid_size=voxel_size,
        sample_inds=sample_inds,
        reduction="center_nearest",
        return_mapping=True,
    )
    
    selected_indices = indices
    
    if isinstance(selected_indices, torch.Tensor):
        selected_indices = selected_indices.cpu().numpy()
    
    return selected_indices, np.array([len(selected_indices)], dtype=np.int64)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def to_tsfm(rot, trans):
    tsfm = np.eye(4)
    tsfm[:3, :3] = rot
    tsfm[:3, 3] = trans.flatten()
    return tsfm


def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None, debug=False, idx_info=None):
    src_pcd_copy = copy.deepcopy(src_pcd)
    src_pcd_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
    
    correspondences = []
    for i, point in enumerate(src_pcd_copy.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if count > 0:
            if K is not None:
                idx = idx[:min(K, count)]
            for j in idx:
                correspondences.append([i, j])
    
    if len(correspondences) == 0:
        correspondences_array = np.zeros((0, 2), dtype=np.int64)
    else:
        correspondences_array = np.array(correspondences, dtype=np.int64)
    
    if debug:
        num_src_points = len(src_pcd.points)
        num_tgt_points = len(tgt_pcd.points)
        num_correspondences = len(correspondences_array)
        
        if num_correspondences > 0:
            unique_src_indices = len(set(correspondences_array[:, 0]))
            unique_tgt_indices = len(set(correspondences_array[:, 1]))
            
            avg_matches_per_src = num_correspondences / unique_src_indices if unique_src_indices > 0 else 0.0
            avg_matches_per_tgt = num_correspondences / unique_tgt_indices if unique_tgt_indices > 0 else 0.0
        else:
            unique_src_indices = 0
            unique_tgt_indices = 0
            avg_matches_per_src = 0.0
            avg_matches_per_tgt = 0.0
        
        unique_match_ratio_src = unique_src_indices / num_src_points if num_src_points > 0 else 0.0
        unique_match_ratio_tgt = unique_tgt_indices / num_tgt_points if num_tgt_points > 0 else 0.0
        
        idx_str = idx_info if idx_info is not None else "unknown"
        logging.info(
            f"[3DMatchNew] Match stats [{idx_str}]: "
            f"total_pairs={num_correspondences}, "
            f"src_points={num_src_points}, "
            f"tgt_points={num_tgt_points}, "
            f"unique_src={unique_src_indices} ({unique_match_ratio_src*100:.2f}%), "
            f"unique_tgt={unique_tgt_indices} ({unique_match_ratio_tgt*100:.2f}%), "
            f"avg_matches_per_src={avg_matches_per_src:.2f}, "
            f"avg_matches_per_tgt={avg_matches_per_tgt:.2f}, "
            f"search_radius={search_voxel_size}"
        )
    
    return correspondences_array


class PairDataset(torch.utils.data.Dataset):
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    self.phase = phase
    self.files = []
    self.data_objects = []
    self.transform = transform
    self.voxel_size = config.voxel_size
    self.matching_search_voxel_size = \
        config.voxel_size * config.positive_pair_search_voxel_size_multiplier

    self.random_scale = random_scale
    self.min_scale = config.min_scale
    self.max_scale = config.max_scale
    self.random_rotation = random_rotation
    self.rotation_range = config.rotation_range
    self.randg = np.random.RandomState()
    if manual_seed:
      self.reset_seed()

  def reset_seed(self, seed=0):
    logging.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

  def __len__(self):
    return len(self.files)


class ThreeDMatchTestDataset(PairDataset):
  DATA_FILES = {
      'test': './config/test_3dmatch.txt'
  }

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               scene_id=None,
               config=None,
               return_ply_names=False):

    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)
    assert phase == 'test', "Supports only the test set."

    self.root = config.threed_match_dir

    subset_names = open(self.DATA_FILES[phase]).read().split()
    if scene_id is not None:
      subset_names = [subset_names[scene_id]]
    for sname in subset_names:
      traj_file = os.path.join(self.root, sname + '-evaluation/gt.log')
      assert os.path.exists(traj_file)
      traj = read_trajectory(traj_file)
      for ctraj in traj:
        i = ctraj.metadata[0]
        j = ctraj.metadata[1]
        T_gt = ctraj.pose
        self.files.append((sname, i, j, T_gt))

    self.return_ply_names = return_ply_names

  def __getitem__(self, pair_index):
    sname, i, j, T_gt = self.files[pair_index]
    ply_name0 = os.path.join(self.root, sname, f'cloud_bin_{i}.ply')
    ply_name1 = os.path.join(self.root, sname, f'cloud_bin_{j}.ply')

    if self.return_ply_names:
      return sname, ply_name0, ply_name1, T_gt

    pcd0 = o3d.io.read_point_cloud(ply_name0)
    pcd1 = o3d.io.read_point_cloud(ply_name1)
    pcd0 = np.asarray(pcd0.points)
    pcd1 = np.asarray(pcd1.points)
    return sname, pcd0, pcd1, T_gt


class IndoorPairDataset(PairDataset):
  OVERLAP_RATIO = None
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)
    self.root = root = config.threed_match_dir
    logging.info(f"Loading the subset {phase} from {root}")

    subset_names = open(self.DATA_FILES[phase]).read().split()
    for name in subset_names:
      fname = name + "*%.2f.txt" % self.OVERLAP_RATIO
      fnames_txt = glob.glob(root + "/" + fname)
      assert len(fnames_txt) > 0, f"Make sure that the path {root} has data {fname}"
      for fname_txt in fnames_txt:
        with open(fname_txt) as f:
          content = f.readlines()
        fnames = [x.strip().split() for x in content]
        for fname in fnames:
          self.files.append([fname[0], fname[1]])

  def __getitem__(self, idx):
    file0 = os.path.join(self.root, self.files[idx][0])
    file1 = os.path.join(self.root, self.files[idx][1])
    data0 = np.load(file0)
    data1 = np.load(file1)
    xyz0 = data0["pcd"]
    xyz1 = data1["pcd"]
    color0 = data0["color"]
    color1 = data1["color"]
    matching_search_voxel_size = self.matching_search_voxel_size

    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1

    if self.random_rotation:
      T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
      T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
      trans = T1 @ np.linalg.inv(T0)

      xyz0 = self.apply_transform(xyz0, T0)
      xyz1 = self.apply_transform(xyz1, T1)
    else:
      trans = np.identity(4)

    # Voxelization
    # Replace ME.utils.sparse_quantize with manual voxel quantization
    def sparse_quantize(coords, return_index=True):
      """Manual sparse quantize for pointcnnpp."""
      coords_int = np.floor(coords).astype(np.int64)
      _, unique_indices = np.unique(coords_int, axis=0, return_index=True)
      if return_index:
        return coords_int[unique_indices], unique_indices
      else:
        return coords_int[unique_indices]
    
    _, sel0 = sparse_quantize(xyz0 / self.voxel_size, return_index=True)
    _, sel1 = sparse_quantize(xyz1 / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0)
    pcd1 = make_open3d_point_cloud(xyz1)

    # Select features and points using the returned voxelized indices
    pcd0.colors = o3d.utility.Vector3dVector(color0[sel0])
    pcd1.colors = o3d.utility.Vector3dVector(color1[sel1])
    pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
    pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points)[sel1])
    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)

    # Get features
    npts0 = len(pcd0.colors)
    npts1 = len(pcd1.colors)

    feats_train0, feats_train1 = [], []

    feats_train0.append(np.ones((npts0, 1)))
    feats_train1.append(np.ones((npts1, 1)))

    feats0 = np.hstack(feats_train0)
    feats1 = np.hstack(feats_train1)

    # Get coords
    xyz0 = np.array(pcd0.points)
    xyz1 = np.array(pcd1.points)

    coords0 = np.floor(xyz0 / self.voxel_size)
    coords1 = np.floor(xyz1 / self.voxel_size)

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    return (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans)


class KITTIPairDataset(PairDataset):
  AUGMENT = None
  DATA_FILES = {
      'train': './config/train_kitti.txt',
      'val': './config/val_kitti.txt',
      'test': './config/test_kitti.txt'
  }
  TEST_RANDOM_ROTATION = False
  IS_ODOMETRY = True

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    # For evaluation, use the odometry dataset training following the 3DFeat eval method
    if self.IS_ODOMETRY:
      self.root = root = config.kitti_root + '/dataset'
      random_rotation = self.TEST_RANDOM_ROTATION
    else:
      self.date = config.kitti_date
      self.root = root = os.path.join(config.kitti_root, self.date)

    self.icp_path = os.path.join(config.kitti_root, 'icp')
    pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)

    logging.info(f"Loading the subset {phase} from {root}")
    # Use the kitti root
    self.max_time_diff = max_time_diff = config.kitti_max_time_diff

    subset_names = open(self.DATA_FILES[phase]).read().split()
    for dirname in subset_names:
      drive_id = int(dirname)
      inames = self.get_all_scan_ids(drive_id)
      for start_time in inames:
        for time_diff in range(2, max_time_diff):
          pair_time = time_diff + start_time
          if pair_time in inames:
            self.files.append((drive_id, start_time, pair_time))

  def get_all_scan_ids(self, drive_id):
    if self.IS_ODOMETRY:
      fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
    else:
      fnames = glob.glob(self.root + '/' + self.date +
                         '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
    assert len(
        fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
    inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
    return inames

  @property
  def velo2cam(self):
    try:
      velo2cam = self._velo2cam
    except AttributeError:
      R = np.array([
          7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
          -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
      ]).reshape(3, 3)
      T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
      velo2cam = np.hstack([R, T])
      self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
    return self._velo2cam

  def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
    if self.IS_ODOMETRY:
      data_path = self.root + '/poses/%02d.txt' % drive
      if data_path not in kitti_cache:
        kitti_cache[data_path] = np.genfromtxt(data_path)
      if return_all:
        return kitti_cache[data_path]
      else:
        return kitti_cache[data_path][indices]
    else:
      data_path = self.root + '/' + self.date + '_drive_%04d_sync/oxts/data' % drive
      odometry = []
      if indices is None:
        fnames = glob.glob(self.root + '/' + self.date +
                           '_drive_%04d_sync/velodyne_points/data/*.bin' % drive)
        indices = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

      for index in indices:
        filename = os.path.join(data_path, '%010d%s' % (index, ext))
        if filename not in kitti_cache:
          kitti_cache[filename] = np.genfromtxt(filename)
        odometry.append(kitti_cache[filename])

      odometry = np.array(odometry)
      return odometry

  def odometry_to_positions(self, odometry):
    if self.IS_ODOMETRY:
      T_w_cam0 = odometry.reshape(3, 4)
      T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
      return T_w_cam0
    else:
      lat, lon, alt, roll, pitch, yaw = odometry.T[:6]

      R = 6378137  # Earth's radius in metres

      # convert to metres
      lat, lon = np.deg2rad(lat), np.deg2rad(lon)
      mx = R * lon * np.cos(lat)
      my = R * lat

      times = odometry.T[-1]
      return np.vstack([mx, my, alt, roll, pitch, yaw, times]).T

  def rot3d(self, axis, angle):
    ei = np.ones(3, dtype='bool')
    ei[axis] = 0
    i = np.nonzero(ei)[0]
    m = np.eye(3)
    c, s = np.cos(angle), np.sin(angle)
    m[i[0], i[0]] = c
    m[i[0], i[1]] = -s
    m[i[1], i[0]] = s
    m[i[1], i[1]] = c
    return m

  def pos_transform(self, pos):
    x, y, z, rx, ry, rz, _ = pos[0]
    RT = np.eye(4)
    RT[:3, :3] = np.dot(np.dot(self.rot3d(0, rx), self.rot3d(1, ry)), self.rot3d(2, rz))
    RT[:3, 3] = [x, y, z]
    return RT

  def get_position_transform(self, pos0, pos1, invert=False):
    T0 = self.pos_transform(pos0)
    T1 = self.pos_transform(pos1)
    return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
        np.linalg.inv(T1), T0).T)

  def _get_velodyne_fn(self, drive, t):
    if self.IS_ODOMETRY:
      fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
    else:
      fname = self.root + \
          '/' + self.date + '_drive_%04d_sync/velodyne_points/data/%010d.bin' % (
              drive, t)
    return fname

  def __getitem__(self, idx):
    drive = self.files[idx][0]
    t0, t1 = self.files[idx][1], self.files[idx][2]
    all_odometry = self.get_video_odometry(drive, [t0, t1])
    positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
    fname0 = self._get_velodyne_fn(drive, t0)
    fname1 = self._get_velodyne_fn(drive, t1)

    # XYZ and reflectance
    xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
    xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

    xyz0 = xyzr0[:, :3]
    xyz1 = xyzr1[:, :3]

    key = '%d_%d_%d' % (drive, t0, t1)
    filename = self.icp_path + '/' + key + '.npy'
    if key not in kitti_icp_cache:
      if not os.path.exists(filename):
        # work on the downsampled xyzs, 0.05m == 5cm
        # Replace ME.utils.sparse_quantize with manual voxel quantization
        def sparse_quantize(coords, return_index=True):
          """Manual sparse quantize for pointcnnpp."""
          coords_int = np.floor(coords).astype(np.int64)
          _, unique_indices = np.unique(coords_int, axis=0, return_index=True)
          if return_index:
            return coords_int[unique_indices], unique_indices
          else:
            return coords_int[unique_indices]
        
        _, sel0 = sparse_quantize(xyz0 / 0.05, return_index=True)
        _, sel1 = sparse_quantize(xyz1 / 0.05, return_index=True)

        M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
             @ np.linalg.inv(self.velo2cam)).T
        xyz0_t = self.apply_transform(xyz0[sel0], M)
        pcd0 = make_open3d_point_cloud(xyz0_t)
        pcd1 = make_open3d_point_cloud(xyz1[sel1])
        # Open3D >= 0.13 uses o3d.pipelines.registration instead of o3d.registration
        try:
          # Try new API first (Open3D >= 0.13)
          reg = o3d.pipelines.registration.registration_icp(
              pcd0, pcd1, 0.2, np.eye(4),
              o3d.pipelines.registration.TransformationEstimationPointToPoint(),
              o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        except AttributeError:
          # Fall back to old API (Open3D < 0.13)
          reg = o3d.registration.registration_icp(
              pcd0, pcd1, 0.2, np.eye(4),
              o3d.registration.TransformationEstimationPointToPoint(),
              o3d.registration.ICPConvergenceCriteria(max_iteration=200))
        pcd0.transform(reg.transformation)
        # pcd0.transform(M2) or self.apply_transform(xyz0, M2)
        M2 = M @ reg.transformation
        # o3d.draw_geometries([pcd0, pcd1])
        # write to a file
        np.save(filename, M2)
      else:
        M2 = np.load(filename)
      kitti_icp_cache[key] = M2
    else:
      M2 = kitti_icp_cache[key]

    if self.random_rotation:
      T0 = sample_random_trans(xyz0, self.randg, np.pi / 4)
      T1 = sample_random_trans(xyz1, self.randg, np.pi / 4)
      trans = T1 @ M2 @ np.linalg.inv(T0)

      xyz0 = self.apply_transform(xyz0, T0)
      xyz1 = self.apply_transform(xyz1, T1)
    else:
      trans = M2

    matching_search_voxel_size = self.matching_search_voxel_size
    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1

    # Voxelization
    xyz0_th = torch.from_numpy(xyz0)
    xyz1_th = torch.from_numpy(xyz1)

    # Replace ME.utils.sparse_quantize with manual voxel quantization
    def sparse_quantize(coords, return_index=True):
      """Manual sparse quantize for pointcnnpp."""
      if isinstance(coords, torch.Tensor):
        coords = coords.numpy()
      coords_int = np.floor(coords).astype(np.int64)
      _, unique_indices = np.unique(coords_int, axis=0, return_index=True)
      if return_index:
        return coords_int[unique_indices], unique_indices
      else:
        return coords_int[unique_indices]
    
    _, sel0 = sparse_quantize(xyz0_th / self.voxel_size, return_index=True)
    _, sel1 = sparse_quantize(xyz1_th / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0[sel0])
    pcd1 = make_open3d_point_cloud(xyz1[sel1])

    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)
    if len(matches) < 1000:
      raise ValueError(f"{drive}, {t0}, {t1}")

    # Get features
    npts0 = len(sel0)
    npts1 = len(sel1)

    feats_train0, feats_train1 = [], []

    unique_xyz0_th = xyz0_th[sel0]
    unique_xyz1_th = xyz1_th[sel1]

    feats_train0.append(torch.ones((npts0, 1)))
    feats_train1.append(torch.ones((npts1, 1)))

    feats0 = torch.cat(feats_train0, 1)
    feats1 = torch.cat(feats_train1, 1)

    coords0 = torch.floor(unique_xyz0_th / self.voxel_size)
    coords1 = torch.floor(unique_xyz1_th / self.voxel_size)

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    return (unique_xyz0_th.float(), unique_xyz1_th.float(), coords0.int(),
            coords1.int(), feats0.float(), feats1.float(), matches, trans)


class KITTINMPairDataset(KITTIPairDataset):
  r"""
  Generate KITTI pairs within N meter distance
  """
  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None,
               pre_downsample_voxel_size=0.3,
               train_info=None,
               val_info=None):
    self.MIN_DIST = config.min_dist
    if self.IS_ODOMETRY:
      self.root = root = os.path.join(config.kitti_root, 'dataset')
      random_rotation = self.TEST_RANDOM_ROTATION
    else:
      self.date = config.kitti_date
      self.root = root = os.path.join(config.kitti_root, self.date)

    self.icp_path = os.path.join(config.kitti_root, 'icp')
    pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)

    logging.info(f"Loading the subset {phase} from {root}")

    subset_names = open(self.DATA_FILES[phase]).read().split()
    if self.IS_ODOMETRY:
      for dirname in subset_names:
        drive_id = int(dirname)
        fnames = glob.glob(root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
        inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])
        # Ensure inames is a Python list, not numpy array
        inames = list(inames)

        all_odo = self.get_video_odometry(drive_id, return_all=True)
        all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
        Ts = all_pos[:, :3, 3]
        pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3))**2
        pdist = np.sqrt(pdist.sum(-1))
        valid_pairs = pdist > self.MIN_DIST
        logging.info(f"  Finding valid pairs (MIN_DIST={self.MIN_DIST})...")
        curr_time = inames[0]
        num_pairs = 0
        while curr_time in inames:
          # Find the min index
          # Note: curr_time is both the time value and the array index (since inames starts from 0)
          next_time = np.where(valid_pairs[curr_time][curr_time:curr_time + 100])[0]
          if len(next_time) == 0:
            curr_time += 1
            continue
          
          # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
          # Convert numpy scalar to Python int
          next_time = int(next_time[0].item() if hasattr(next_time[0], 'item') else int(next_time[0])) + curr_time - 1
          # Ensure it's a Python int
          next_time = int(next_time)

          if next_time in inames:
            self.files.append((drive_id, curr_time, next_time))
            num_pairs += 1
            curr_time = next_time + 1
        logging.info(f"  Found {num_pairs} valid pairs for drive {drive_id}")
    else:
      for dirname in subset_names:
        drive_id = int(dirname)
        fnames = glob.glob(root + '/' + self.date +
                           '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
        assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
        inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

        all_odo = self.get_video_odometry(drive_id, return_all=True)
        all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
        Ts = all_pos[:, 0, :3]

        pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3))**2
        pdist = np.sqrt(pdist.sum(-1))

        for start_time in inames:
          pair_time = np.where(
              pdist[start_time][start_time:start_time + 100] > self.MIN_DIST)[0]
          if len(pair_time) == 0:
            continue
          else:
            # Convert numpy scalar to Python int
            pair_time = int(pair_time[0].item() if hasattr(pair_time[0], 'item') else int(pair_time[0])) + start_time
            # Ensure it's a Python int
            pair_time = int(pair_time)

          if pair_time in inames:
            self.files.append((drive_id, start_time, pair_time))

    if self.IS_ODOMETRY:
      # Remove problematic sequence
      for item in [
          (8, 15, 58),
      ]:
        if item in self.files:
          self.files.pop(self.files.index(item))


class ThreeDMatchNewPairDatasetPure(PairDataset):
    
    def __init__(self, phase, transform=None, random_scale=False, random_rotation=False, 
                 manual_seed=False, config=None, pre_downsample_voxel_size=None, 
                 train_info=None, val_info=None, data_augmentation=True):
        PairDataset.__init__(self, phase, transform, random_rotation, 
                           random_scale, manual_seed, config)
        
        self.config = config
        
        self.pre_downsample_voxel_size = pre_downsample_voxel_size
        self.phase = phase
        self.data_augmentation = data_augmentation
        
        self.threedmatch_root = (
            getattr(config, 'threedmatch_root', None) or
            getattr(config, 'threed_match_dir', None) or
            os.environ.get('THREEDMATCH_ROOT', '')
        )
        if not self.threedmatch_root:
            raise ValueError(
                "3DMatch root not set. Set THREEDMATCH_ROOT env var or pass "
                "--threedmatch_root / --threed_match_dir (e.g. export THREEDMATCH_ROOT=/path/to/3dmatch_processed/indoor)"
            )
        self.threedmatch_train_dir = os.path.join(self.threedmatch_root, phase)
        
        self.overlap_radius = 1.5 * getattr(config, 'pre_downsample_voxel_size', 0.02)
        self.augment_noise = getattr(config, 'augment_noise', 0.005)
        self.rot_factor = getattr(config, 'rot_factor', 1.0)
        
        if hasattr(config, 'use_random_scale'):
            self.random_scale = config.use_random_scale
        if hasattr(config, 'use_random_rotation'):
            self.random_rotation = config.use_random_rotation
        
        self.jitter_sigma = getattr(config, 'jitter_sigma', 0.01)
        
        self.debug_correspondences = False
        
        self.root = self.threedmatch_train_dir
        
        use_pkl = False
        _fcgf_configs = os.path.join(os.path.dirname(__file__), '..', 'configs', 'indoor')
        default_train_info = os.path.join(_fcgf_configs, 'train_info.pkl')
        default_val_info = os.path.join(_fcgf_configs, 'val_info.pkl')
        
        if phase == 'train':
            if train_info is not None and os.path.exists(train_info):
                use_pkl = True
                info_file = train_info
            elif hasattr(config, 'train_info') and os.path.exists(config.train_info):
                use_pkl = True
                info_file = config.train_info
            elif os.path.exists(default_train_info):
                use_pkl = True
                info_file = default_train_info
        elif phase == 'test':
            if train_info is not None and os.path.exists(train_info):
                use_pkl = True
                info_file = train_info
            elif hasattr(config, 'train_info') and os.path.exists(config.train_info):
                use_pkl = True
                info_file = config.train_info
            elif os.path.exists(default_train_info):
                use_pkl = True
                info_file = default_train_info
            elif val_info is not None and os.path.exists(val_info):
                use_pkl = True
                info_file = val_info
            elif hasattr(config, 'val_info') and os.path.exists(config.val_info):
                use_pkl = True
                info_file = config.val_info
            elif os.path.exists(default_val_info):
                use_pkl = True
                info_file = default_val_info
        elif phase == 'val':
            if val_info is not None and os.path.exists(val_info):
                use_pkl = True
                info_file = val_info
            elif hasattr(config, 'val_info') and os.path.exists(config.val_info):
                use_pkl = True
                info_file = config.val_info
            elif os.path.exists(default_val_info):
                use_pkl = True
                info_file = default_val_info
        
        if use_pkl:
            logging.info(f"[3DMatchNew] Loading pairs from pkl file: {info_file}")
            self.infos = load_obj(info_file)
            self.use_pkl = True
            
            if not isinstance(self.infos, dict) or 'rot' not in self.infos:
                raise ValueError(f"Invalid pkl file format, should contain 'rot', 'trans', 'src', 'tgt' keys")
            
            self.base_dir = self.threedmatch_root
            logging.info(f"[3DMatchNew] Using 3DMatch data directory as base_dir: {self.base_dir}")
            
            logging.info(f"[3DMatchNew] Loaded {len(self.infos['rot'])} pairs")
            logging.info(f"[3DMatchNew] base_dir: {self.base_dir}")
        else:
            raise ValueError(
                f"[3DMatchNew] Auto-scanning directory mode is not supported. "
                f"Please provide pkl file via train_info or val_info parameter, "
                f"or set config.train_info/config.val_info. "
                f"Phase: {phase}"
            )
        
        logging.info(f"[3DMatchNew] overlap_radius={self.overlap_radius}, augment_noise={self.augment_noise}")
    
    def __str__(self):
        return "ThreeDMatchNewPairDatasetPure"
    
    def __len__(self):
        if self.use_pkl:
            return len(self.infos['rot'])
        else:
            return len(self.files)
    
    def _load_point_cloud_from_file(self, file_path):
        original_path = file_path
        ply_path = None
        pth_path = None
        
        if file_path.endswith('.ply'):
            ply_path = file_path
            pth_path = file_path.replace('.ply', '.pth')
        elif file_path.endswith('.pth'):
            pth_path = file_path
            ply_path = file_path.replace('.pth', '.ply')
        else:
            ply_path = file_path + '.ply'
            pth_path = file_path + '.pth'
        
        if pth_path and os.path.exists(pth_path):
            try:
                pcd_data = torch.load(pth_path, weights_only=False)
                
                if isinstance(pcd_data, np.ndarray):
                    xyz = pcd_data.astype(np.float32)
                elif isinstance(pcd_data, torch.Tensor):
                    xyz = pcd_data.cpu().numpy().astype(np.float32)
                else:
                    if hasattr(pcd_data, 'points'):
                        xyz = np.asarray(pcd_data.points).astype(np.float32)
                    elif isinstance(pcd_data, dict) and 'points' in pcd_data:
                        xyz = np.asarray(pcd_data['points']).astype(np.float32)
                    else:
                        raise ValueError(f"Unrecognized .pth file format: {type(pcd_data)}")
                
                if xyz.ndim != 2 or xyz.shape[1] != 3:
                    raise ValueError(f"Invalid point cloud data format: shape={xyz.shape}, expected [N, 3]")
                
                if len(xyz) == 0:
                    raise ValueError(f"Empty point cloud file: {pth_path}")
                
                color = np.ones((len(xyz), 3), dtype=np.float32)
                return xyz, color
            except Exception as e:
                logging.warning(f"Failed to read .pth file: {pth_path}, error: {e}, trying .ply file")
        
        if ply_path and os.path.exists(ply_path):
            try:
                pcd_o3d = o3d.io.read_point_cloud(ply_path)
                xyz = np.asarray(pcd_o3d.points).astype(np.float32)
                
                if len(xyz) == 0:
                    raise ValueError(f"Empty point cloud file: {ply_path}")
                
                if pcd_o3d.has_colors():
                    color = np.asarray(pcd_o3d.colors).astype(np.float32)
                else:
                    color = np.ones((len(xyz), 3), dtype=np.float32)
                
                return xyz, color
            except Exception as e:
                raise RuntimeError(f"Failed to read .ply file: {ply_path}, error: {e}")
        
        raise FileNotFoundError(
            f"Point cloud file not found: {original_path}\n"
            f"  Tried .pth path: {pth_path} (exists: {os.path.exists(pth_path) if pth_path else False})\n"
            f"  Tried .ply path: {ply_path} (exists: {os.path.exists(ply_path) if ply_path else False})"
        )
    
    def _load_pose(self, scene_name, fragment_id):
        pose_file = os.path.join(self.root, scene_name, 'poses', f'cloud_bin_{fragment_id}.txt')
        if not os.path.exists(pose_file):
            return None
        
        try:
            with open(pose_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 5:
                logging.warning(f"Invalid pose file format: {pose_file}")
                return None
            
            trans = np.zeros((4, 4))
            for i in range(4):
                row = [float(x) for x in lines[i+1].strip().split()]
                trans[i, :] = row
            
            return trans
        except Exception as e:
            logging.warning(f"Failed to read pose file: {pose_file}, error: {e}")
            return None
    
    def _get_fragment_id_from_filename(self, filename):
        basename = os.path.basename(filename)
        id_str = basename.replace('cloud_bin_', '').replace('.pth', '').replace('.ply', '')
        return int(id_str)
    
    def _get_fragment_fn(self, scene_name, fragment_id):
        fragments_dir = os.path.join(self.root, scene_name, 'fragments')
        scene_dir = os.path.join(self.root, scene_name)
        
        if os.path.exists(fragments_dir):
            base_dir = fragments_dir
        else:
            base_dir = scene_dir
        
        pth_path = os.path.join(base_dir, f'cloud_bin_{fragment_id}.pth')
        if os.path.exists(pth_path):
            return pth_path
        ply_path = os.path.join(base_dir, f'cloud_bin_{fragment_id}.ply')
        return ply_path
    
    def get_all_fragments(self, scene_name):
        fragments_dir = os.path.join(self.root, scene_name, 'fragments')
        scene_dir = os.path.join(self.root, scene_name)
        
        if os.path.exists(fragments_dir):
            search_dir = fragments_dir
        elif os.path.exists(scene_dir):
            search_dir = scene_dir
        else:
            logging.warning(f"Scene directory does not exist: {scene_dir}")
            return []
        
        pth_files = sorted(glob.glob(os.path.join(search_dir, 'cloud_bin_*.pth')))
        if len(pth_files) > 0:
            fragment_ids = [self._get_fragment_id_from_filename(fname) for fname in pth_files]
        else:
            ply_files = sorted(glob.glob(os.path.join(search_dir, 'cloud_bin_*.ply')))
            fragment_ids = [self._get_fragment_id_from_filename(fname) for fname in ply_files]
        
        return sorted(fragment_ids)
    
    def generate_pairs(self, scene_filter=None):
        files = []
        
        logging.info(f"Loading the subset {self.phase} from {self.root}")
        
        if not os.path.exists(self.root):
            logging.error(f"3DMatch data directory does not exist: {self.root}")
            return []
        
        scene_dirs = [d for d in os.listdir(self.root) 
                     if os.path.isdir(os.path.join(self.root, d))]
        
        if scene_filter is not None:
            scene_dirs = [d for d in scene_dirs if d in scene_filter]
            logging.info(f"Using scene filter, filtered {len(scene_dirs)} scenes from {len(scene_filter)} scenes")
        
        self.scenes = sorted(scene_dirs)
        logging.info(f"Found {len(self.scenes)} scenes")
        
        for scene_name in self.scenes:
            fragment_ids = self.get_all_fragments(scene_name)
            
            if len(fragment_ids) == 0:
                logging.warning(f"Scene {scene_name} has no fragment data")
                continue
            
            logging.info(f"Scene {scene_name}: found {len(fragment_ids)} fragments")
            
            for i, frag_id0 in enumerate(fragment_ids):
                if self.max_time_diff is not None:
                    max_idx = min(i + self.max_time_diff + 1, len(fragment_ids))
                    candidates = fragment_ids[i+1:max_idx]
                else:
                    candidates = fragment_ids[i+1:]
                
                for frag_id1 in candidates:
                    pose0 = self._load_pose(scene_name, frag_id0)
                    pose1 = self._load_pose(scene_name, frag_id1)
                    
                    if pose0 is not None and pose1 is not None:
                        M_rel = np.linalg.inv(pose0) @ pose1
                        distance = np.linalg.norm(M_rel[:3, 3])
                        
                        if distance > self.max_pair_distance:
                            continue
                    
                    files.append((scene_name, frag_id0, frag_id1))
        
        logging.info(f"Generated {len(files)} pairs")
        return files
    
    def __getitem__(self, idx):
        if self.use_pkl:
            rot = self.infos['rot'][idx]
            trans = self.infos['trans'][idx]
            
            src_path_pkl = self.infos['src'][idx]
            tgt_path_pkl = self.infos['tgt'][idx]
            
            src_path = os.path.join(self.base_dir, src_path_pkl)
            tgt_path = os.path.join(self.base_dir, tgt_path_pkl)

            try:
                src_pcd, _ = self._load_point_cloud_from_file(src_path)
                tgt_pcd, _ = self._load_point_cloud_from_file(tgt_path)
            except Exception as e:
                error_msg = f"Failed to load point cloud: src_path={src_path}, tgt_path={tgt_path}, error={e}"
                logging.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from e
        else:
            scene_name, frag_id0, frag_id1 = self.files[idx]
            
            src_path = self._get_fragment_fn(scene_name, frag_id0)
            tgt_path = self._get_fragment_fn(scene_name, frag_id1)
            
            src_pcd, _ = self._load_point_cloud_from_file(src_path)
            tgt_pcd, _ = self._load_point_cloud_from_file(tgt_path)
            
            pose0 = self._load_pose(scene_name, frag_id0)
            pose1 = self._load_pose(scene_name, frag_id1)
            
            if pose0 is None or pose1 is None:
                logging.warning(f"Cannot get pose, using identity transform: {scene_name}, {frag_id0}, {frag_id1}")
                rot = np.eye(3)
                trans = np.zeros(3)
            else:
                M_rel = np.linalg.inv(pose0) @ pose1
                rot = M_rel[:3, :3]
                trans = M_rel[:3, 3]
        
        if len(src_pcd) == 0 or len(tgt_pcd) == 0:
            if self.use_pkl:
                idx_info = f"idx={idx}"
            else:
                scene_name, frag_id0, frag_id1 = self.files[idx]
                idx_info = f"scene={scene_name}, frag0={frag_id0}, frag1={frag_id1}"
            error_msg = f"Empty point cloud: src={len(src_pcd)}, tgt={len(tgt_pcd)} [{idx_info}]"
            logging.warning(f"⚠️  {error_msg}")
            raise ValueError(error_msg)
        
        matching_search_voxel_size = self.overlap_radius
        
        if self.data_augmentation and self.phase in ['train']:
            if self.random_scale and random.random() < 0.95:
                scale = self.min_scale + (self.max_scale - self.min_scale) * random.random()
                matching_search_voxel_size *= scale
                src_pcd = scale * src_pcd
                tgt_pcd = scale * tgt_pcd
            
            if self.random_rotation:
                T0 = sample_random_trans(src_pcd, self.randg, self.rotation_range)
                T1 = sample_random_trans(tgt_pcd, self.randg, self.rotation_range)
                trans_rel = T1 @ np.linalg.inv(T0)
                
                src_pcd = self.apply_transform(src_pcd, T0)
                tgt_pcd = self.apply_transform(tgt_pcd, T1)
                
                rot = trans_rel[:3, :3]
                trans = trans_rel[:3, 3]
            
            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise
        
        if trans.ndim == 1:
            trans = trans[:, None]
        
        tsfm = to_tsfm(rot, trans)
        
        sel0, new_batch_list0_final = voxel_downsample_using_pointcloud_downsampler(src_pcd, self.pre_downsample_voxel_size)
        sel1, new_batch_list1_final = voxel_downsample_using_pointcloud_downsampler(tgt_pcd, self.pre_downsample_voxel_size)
        
        src_pcd = src_pcd[sel0]
        tgt_pcd = tgt_pcd[sel1]

        min_points = 100
        if len(src_pcd) < min_points or len(tgt_pcd) < min_points:
            if self.use_pkl:
                idx_info = f"idx={idx}"
            else:
                scene_name, frag_id0, frag_id1 = self.files[idx]
                idx_info = f"scene={scene_name}, frag0={frag_id0}, frag1={frag_id1}"
            error_msg = f"Too few points: src={len(src_pcd)}, tgt={len(tgt_pcd)} (need at least {min_points} points) [{idx_info}]"
            logging.warning(f"⚠️  {error_msg}")
            raise ValueError(error_msg)
        
        src_pcd_o3d = make_open3d_point_cloud(src_pcd)
        tgt_pcd_o3d = make_open3d_point_cloud(tgt_pcd)
        
        if self.debug_correspondences:
            if self.use_pkl:
                idx_info_str = f"idx={idx}"
            else:
                scene_name, frag_id0, frag_id1 = self.files[idx]
                idx_info_str = f"scene={scene_name}, frag0={frag_id0}, frag1={frag_id1}"
        else:
            idx_info_str = None
        
        if self.data_augmentation and self.phase in ['train'] and self.random_scale:
            current_overlap_radius = matching_search_voxel_size
        else:
            current_overlap_radius = self.overlap_radius
        correspondences = get_correspondences(
            src_pcd_o3d, 
            tgt_pcd_o3d, 
            tsfm, 
            current_overlap_radius,
            debug=self.debug_correspondences,
            idx_info=idx_info_str if self.debug_correspondences else None
        )
        
        try:
            num_src_points = src_pcd.shape[0]
            num_tgt_points = tgt_pcd.shape[0]
            corr_arr = correspondences if isinstance(correspondences, np.ndarray) else np.array(correspondences, dtype=np.int64)
            if corr_arr.shape[0] > 0:
                unique_src = len(set(corr_arr[:, 0].tolist()))
                unique_tgt = len(set(corr_arr[:, 1].tolist()))
            else:
                unique_src = 0
                unique_tgt = 0
            overlap_src = (unique_src / num_src_points) if num_src_points > 0 else 0.0
            overlap_tgt = (unique_tgt / num_tgt_points) if num_tgt_points > 0 else 0.0
            overlap_min = min(overlap_src, overlap_tgt)
            if self.use_pkl:
                pair_info_str = f"idx={idx}"
            else:
                scene_name, frag_id0, frag_id1 = self.files[idx]
                pair_info_str = f"{scene_name}/({frag_id0},{frag_id1})"
            if overlap_src < 0.3 or overlap_tgt < 0.3:
                if len(self) > 1:
                    next_idx = (idx + 1) % len(self)
                    return self.__getitem__(next_idx)
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            else:
                pass
        
        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        
        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)
        
        if self.data_augmentation and self.phase in ['train']:
            if random.random() < 0.95:
                src_feats += np.random.normal(0, self.jitter_sigma, src_feats.shape).astype(np.float32)
                tgt_feats += np.random.normal(0, self.jitter_sigma, tgt_feats.shape).astype(np.float32)
        
        if self.transform:
            coords0_dummy = np.zeros((len(src_feats), 3))
            coords1_dummy = np.zeros((len(tgt_feats), 3))
            _, src_feats = self.transform(coords0_dummy, src_feats)
            _, tgt_feats = self.transform(coords1_dummy, tgt_feats)
        
        matches_list = correspondences.tolist() if len(correspondences) > 0 else []
        
        new_batch_list0_final = torch.tensor([len(src_pcd)], dtype=torch.long)
        new_batch_list1_final = torch.tensor([len(tgt_pcd)], dtype=torch.long)
        
        if self.use_pkl:
            pair_info = (src_path_pkl, tgt_path_pkl)
        else:
            scene_name, frag_id0, frag_id1 = self.files[idx]
            pair_info = (scene_name, frag_id0, frag_id1)
        return (src_pcd, tgt_pcd, src_feats, tgt_feats,
                matches_list, tsfm, new_batch_list0_final, new_batch_list1_final, pair_info)


class ThreeDMatchPairDataset(IndoorPairDataset):
  OVERLAP_RATIO = 0.3
  DATA_FILES = {
      'train': '/data3/lihan/registration/FCGF_modified/configs/indoor/train_3dmatch.txt',
      'val': '/data3/lihan/registration/FCGF_modified/configs/indoor/val_3dmatch.txt',
      'test': '/data3/lihan/registration/FCGF_modified/configs/indoor/train_3dmatch.txt'
  }


ALL_DATASETS = [ThreeDMatchNewPairDatasetPure, ThreeDMatchPairDataset, KITTIPairDataset, KITTINMPairDataset]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, phase, batch_size, num_threads=0, shuffle=None):
  assert phase in ['train', 'trainval', 'val', 'test']
  if shuffle is None:
    shuffle = phase != 'test'

  if config.dataset not in dataset_str_mapping.keys():
    logging.error(f'Dataset {config.dataset}, does not exists in ' +
                  ', '.join(dataset_str_mapping.keys()))

  Dataset = dataset_str_mapping[config.dataset]

  use_random_scale = False
  use_random_rotation = False
  transforms = []
  if phase in ['train', 'trainval']:
    use_random_rotation = config.use_random_rotation
    use_random_scale = config.use_random_scale
    transforms += [t.Jitter()]

  train_info = getattr(config, 'train_info', None)
  val_info = getattr(config, 'val_info', None)
  
  pre_downsample_voxel_size = getattr(config, 'pre_downsample_voxel_size', None)
  
  dset = Dataset(
      phase,
      transform=t.Compose(transforms),
      random_scale=use_random_scale,
      random_rotation=use_random_rotation,
      config=config,
      pre_downsample_voxel_size=pre_downsample_voxel_size,
      train_info=train_info,
      val_info=val_info)

  collate_fn = make_collate_pair_fn(config.voxel_size)
  
  if len(dset) == 0:
      dataset_name = Dataset.__name__
      logging.error(f"Dataset {dataset_name} is empty in {phase} phase, skipping DataLoader creation")
      
      if dataset_name == 'ThreeDMatchNewPairDatasetPure':
          if hasattr(dset, 'use_pkl'):
              if dset.use_pkl:
                  logging.error(f"   - Using PKL file mode, but pair count is 0")
                  if hasattr(dset, 'base_dir'):
                      logging.error(f"   - base_dir: {dset.base_dir}")
              else:
                  logging.error(f"   - Using auto-scan mode, but no pairs found")
                  if hasattr(dset, 'root'):
                      logging.error(f"   - Data directory: {dset.root}")
                      logging.error(f"   - Data directory exists: {os.path.exists(dset.root)}")
                  if hasattr(dset, 'scenes'):
                      logging.error(f"   - Found {len(dset.scenes)} scenes")
                      if len(dset.scenes) > 0:
                          logging.error(f"   - Scene list: {dset.scenes[:5]}")
      
      train_info = getattr(config, 'train_info', None)
      val_info = getattr(config, 'val_info', None)
      threedmatch_root = getattr(config, 'threedmatch_root', None)
      logging.error(f"   - config.train_info: {train_info}")
      logging.error(f"   - config.val_info: {val_info}")
      logging.error(f"   - config.threedmatch_root: {threedmatch_root}")
      
      if phase == 'train':
          logging.error("   Training dataset is empty, cannot continue training!")
      return None
  
  loader_kwargs = {
      'batch_size': batch_size,
      'shuffle': shuffle,
      'num_workers': num_threads,
      'collate_fn': collate_fn,
      'pin_memory': True,
      'drop_last': True if phase == 'train' else False,
  }
  if num_threads > 0:
      loader_kwargs['persistent_workers'] = True
  
  loader = torch.utils.data.DataLoader(dset, **loader_kwargs)

  return loader
