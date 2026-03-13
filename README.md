# PointCNN++ for FCGF Backbone

This module implements **Fully Convolutional Geometric Features (FCGF)** with **PointCNN++** as the backbone instead of MinkowskiEngine. It is used for 3D geometric feature extraction, registration, and related tasks (e.g. 3DMatch, KITTI). This document describes how to install dependencies, prepare data, and run training and benchmarking.

---

## Dependencies

### 1. Environment and base requirements

- **Python**: 3.8 or higher (3.10 recommended).
- **CUDA**: Match your PyTorch version (e.g. CUDA 12.1+).
- **PyTorch**: 2.6 or higher.

### 2. Install project dependencies

From this FCGF module directory (or the Pointcept repo root with `PYTHONPATH` set):

```bash
cd /path/to/Pointcept/pointcept/models/FCGF
pip install -r requirements.txt
```

`requirements.txt` includes `numpy`, `scipy`, `matplotlib`, `open3d`, `tensorboardX`, `easydict`, `joblib`, `scikit-learn`, etc. Install PyTorch separately (e.g. via conda or [pytorch.org](https://pytorch.org)).

---

## Notes

- **Data paths**: 3DMatch root is set via `THREEDMATCH_ROOT` (default in the script: `path/to/3dmatch_processed/indoor`). KITTI root is set via `KITTI_PATH`. Override them for your environment.
- **Output directory**: Training outputs (checkpoints, logs) go under `DATA_ROOT` (default `./outputs/Experiments`) with a dataset/trainer/model/time subfolder.
- **Voxel size**: 3DMatch default is `0.025` (2.5 cm); KITTI default is `0.3` (30 cm). Set `VOXEL_SIZE` in the script or environment if needed.
- **Model**: Default backbone is `ResUNetBN2C` (PointCNN++-based ResUNet). Other variants (e.g. `ResUNetBN2B`, `ResUNetBN2D`) are in `model/resunet.py`.

---

## Steps overview

1. Install the dependencies above (including PointCNN++).
2. Prepare 3DMatch and/or KITTI data (see Data preparation).
3. Run training for 3DMatch and/or KITTI (see Training).
4. (Optional) Run the demo and/or 3DMatch benchmark (see Demo and Benchmark).

---

## Data preparation

### 3DMatch

Follow the same 3DMatch data preparation as [OverlapPredator](https://github.com/prs-eth/OverlapPredator): download preprocessed pairwise datasets or raw dense data via their scripts, then organize the folder as `train/` (with scene subdirs containing `fragments/` and `poses/`), `test/`, and place `train_info.pkl` / `val_info.pkl` under `THREEDMATCH_ROOT` (or use the FCGF `configs/indoor/` pkl files). See [OverlapPredator README](https://github.com/prs-eth/OverlapPredator) for details.

Or, you can straightly download the processed dataset with link: `https://drive.google.com/file/d/1zsZbJSID5AL4diJuhC0gZDYJsz-PidhH/view?usp=sharing`

### KITTI Odometry

- Download the [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) training set and set `KITTI_PATH` to the root that contains `dataset/sequences`.

---

## Pretrained Weights

Pre-trained checkpoints for the PointCNN++ backbone (ResUNetBN2C) are available:

| Dataset | URL |
|---------|-----|
| **KITTI** | [Google Drive](https://drive.google.com/file/d/12ahfWCwJyaCJwcgqlKDgK-sqPCaTJtif/view?usp=drive_link) |
| **3DMatch** | [Google Drive](https://drive.google.com/file/d/1Wkyb9QSyKsTYPErUOex6lbMwkXootIFk/view?usp=sharing) |

Download the checkpoint and place it (e.g. as `best_val_checkpoint.pth`) in your model directory. Use it when running the demo or benchmark script with `--resume` or `--checkpoint` (see the script for the exact argument).

---

## Training

### 3DMatch (PointCNN++ backbone)

**Custom arguments** (environment variables or script edits):

- `THREEDMATCH_ROOT`: 3DMatch data root.
- `VOXEL_SIZE`: e.g. `0.025` or `0.05`.
- `MODEL_N_OUT`: feature dimension (e.g. 16 or 32).
- `BATCH_SIZE`, `MAX_EPOCH`, `LR`, etc.

Example with custom root and voxel size:

```bash
export THREEDMATCH_ROOT=/path/to/3dmatch_processed/indoor
export CONFIG_PATH=./configs/indoor
export VOXEL_SIZE=0.025
bash ./scripts/train_3dmatch.sh
```

### KITTI (PointCNN++ backbone)

```bash
export KITTI_PATH=/path/to/kitti_odometry_dataset
bash ./scripts/train_kitti.sh
```

Defaults: dataset `KITTINMPairDataset`, model `ResUNetBN2C`, voxel size `0.3`, feature dimension 16. Logs and checkpoints go to `./outputs/Experiments/...`.

---

## Config and script mapping

| Task           | Script | Main env vars | Description |
|----------------|--------|----------------|-------------|
| 3DMatch train  | `scripts/train_3dmatch.sh` | `THREEDMATCH_ROOT`, `VOXEL_SIZE`, `MODEL`, `MODEL_N_OUT` | Train FCGF (PointCNN++ backbone) on 3DMatch |
| KITTI train    | `scripts/train_kitti.sh`   | `KITTI_PATH`, `VOXEL_SIZE`, `MODEL`, `MODEL_N_OUT`       | Train FCGF on KITTI |
---

## Reference: FCGF (ICCV 2019)

The original FCGF uses MinkowskiEngine and achieves state-of-the-art accuracy for 3D feature matching. This module keeps the same FCGF design (fully convolutional metric learning, hardest contrastive/triplet loss) but replaces the backbone with PointCNN++.

- [ICCV'19 Paper](https://node1.chrischoy.org/data/publications/fcgf/fcgf.pdf)
- Citation:
  ```bibtex
  @inproceedings{FCGF2019,
      author = {Christopher Choy and Jaesik Park and Vladlen Koltun},
      title = {Fully Convolutional Geometric Features},
      booktitle = {ICCV},
      year = {2019},
  }
  ```

The Model Zoo table in the original README refers to MinkowskiEngine checkpoints; for the PointCNN++ backbone, train with the scripts above and use the resulting checkpoints for demo and benchmark.
