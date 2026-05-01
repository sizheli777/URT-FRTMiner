"""
Strip-based 序列化 — 将无序点云转为有序序列供 TCN/BiLSTM 处理。

核心思路:
  1. 对每个聚类块做 PCA → 确定法线方向（横跨边界的方向）
  2. 沿法线方向等距切成 N 个条带 (strip)
  3. 每个 strip 内聚合特征 → 得到 (N, F) 有序序列
  4. 序列模型沿 strip 方向推理 → 捕捉 田→边界→路 的空间模式
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label as connected_label
from typing import Tuple, Optional, List, Dict
import torch
from torch.utils.data import Dataset


# ═══════════════════════════════════════════════════════════════════
# Part 1: 密度聚类 — 发现田块 / 道路区域
# ═══════════════════════════════════════════════════════════════════

def density_cluster(
    coords_m: np.ndarray,
    grid_size: float = 20.0,
    density_factor: float = 1.5,
    min_points: int = 30,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    基于网格密度的空间聚类, 将点云分为高密度区(田块候选) 和 低密度区(道路候选)。

    Returns:
        labels: (N,) 聚类标签, -1=低密度(未分配), >=0=高密度区ID
        density_grid: (ny, nx) 密度网格
        info: 聚类统计信息
    """
    n = len(coords_m)
    if n == 0:
        return np.full(n, -1, dtype=np.int32), np.zeros((1,1)), {}

    x, y = coords_m[:, 0], coords_m[:, 1]
    grid_size = float(grid_size)

    x_min, x_max = x.min() - grid_size, x.max() + grid_size
    y_min, y_max = y.min() - grid_size, y.max() + grid_size
    nx = max(5, int((x_max - x_min) / grid_size))
    ny = max(5, int((y_max - y_min) / grid_size))

    density_grid = np.zeros((ny, nx), dtype=np.float32)
    xi = np.clip(((x - x_min) / grid_size).astype(np.int32), 0, nx - 1)
    yi = np.clip(((y - y_min) / grid_size).astype(np.int32), 0, ny - 1)

    for px, py in zip(xi, yi):
        density_grid[py, px] += 1

    # 密度阈值
    density_flat = density_grid[density_grid > 0]
    density_threshold = np.median(density_flat) * density_factor if len(density_flat) > 0 else 5

    # 连通高密度区
    high_density = density_grid >= density_threshold
    labeled_regions, num_regions = connected_label(high_density)

    # 分配每个点到聚类
    labels = np.full(n, -1, dtype=np.int32)
    for region_id in range(1, num_regions + 1):
        region_mask = labeled_regions == region_id
        region_y, region_x = np.where(region_mask)
        region_points = []
        for gy, gx in zip(region_y, region_x):
            in_cell = (yi == gy) & (xi == gx)
            region_points.append(np.where(in_cell)[0])
        if region_points:
            indices = np.concatenate(region_points)
            if len(indices) >= min_points:
                labels[indices] = region_id - 1

    info = {
        "grid_shape": (ny, nx),
        "density_threshold": density_threshold,
        "num_regions": num_regions,
        "num_clustered": int((labels >= 0).sum()),
    }
    return labels, density_grid, info, (x_min, x_max, y_min, y_max)


# ═══════════════════════════════════════════════════════════════════
# Part 2: PCA方向计算 + Strip切分
# ═══════════════════════════════════════════════════════════════════

def compute_pca_direction(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对点集做PCA, 返回主方向和法线方向。

    Returns:
        main_dir: 第一主成分 → 沿路方向 (延伸方向)
        normal_dir: 第二主成分 → 横跨边界方向 (切片方向)
        eigvals: 特征值 [λ_small, λ_large]
    """
    centered = points - points.mean(axis=0)
    if len(centered) < 3:
        # 点太少, 默认水平/竖直
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])

    cov = centered.T @ centered / (len(centered) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvals 升序: [λ_min, λ_max]
    normal_dir = eigvecs[:, 0]   # 最小特征值 → 法线(切片方向)
    main_dir = eigvecs[:, 1]     # 最大特征值 → 沿路方向
    return main_dir, normal_dir, eigvals


def cluster_to_strip_sequence(
    cluster_points: np.ndarray,        # (M, 2) 米制坐标
    cluster_speed: np.ndarray,         # (M,) 速度
    cluster_direction: np.ndarray,     # (M,) 方向
    cluster_probs: Optional[np.ndarray] = None,  # (M,) 模型概率 (可选)
    n_strips: int = 64,
    strip_width: Optional[float] = None,  # 等距条带宽度(米), None=自适应
    min_points_per_strip: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    将聚类块沿PCA法线方向切成N个strip, 每个strip聚合成一个特征向量。

    Args:
        cluster_points: (M, 2) 米制坐标
        cluster_speed: (M,) 速度
        cluster_direction: (M,) 方向
        cluster_probs: (M,) 可选, 模型预测概率
        n_strips: strip数量
        strip_width: strip宽度, None=自适应
        min_points_per_strip: 每strip最少点数

    Returns:
        sequence:     (n_strips, F) 特征序列
        validity:     (n_strips,) 有效mask (点数>=min_points)
        strip_centers: (n_strips, 2) 每个strip的中心坐标
        info:         调试信息
    """
    M = len(cluster_points)
    if M < n_strips:
        n_strips = max(4, M // 2)

    main_dir, normal_dir, eigvals = compute_pca_direction(cluster_points)
    centered = cluster_points - cluster_points.mean(axis=0)
    proj = centered @ normal_dir  # 沿法线方向的投影值

    # 等距分箱
    proj_min, proj_max = proj.min(), proj.max()
    if strip_width is not None:
        n_strips = max(4, int((proj_max - proj_min) / strip_width))
    bins = np.linspace(proj_min - 1e-6, proj_max + 1e-6, n_strips + 1)

    # 聚合每个strip
    F = 10  # 特征维度
    sequence = np.zeros((n_strips, F), dtype=np.float32)
    validity = np.zeros(n_strips, dtype=bool)
    strip_centers = np.zeros((n_strips, 2), dtype=np.float32)
    strip_labels = np.zeros(n_strips, dtype=np.float32)  # 平均标签(如果有)

    # 归一化用的全局统计
    speed_mean = np.mean(cluster_speed) if len(cluster_speed) > 0 else 0.0
    speed_std = np.std(cluster_speed) if len(cluster_speed) > 0 else 1.0

    for i in range(n_strips):
        mask = (proj >= bins[i]) & (proj < bins[i + 1])
        n_pts = mask.sum()

        strip_centers[i] = cluster_points[mask].mean(axis=0) if n_pts > 0 else [0, 0]

        if n_pts >= min_points_per_strip:
            validity[i] = True
            pts_i = cluster_points[mask]
            spd_i = cluster_speed[mask]
            dir_i = cluster_direction[mask]

            # 特征设计:
            # [0] 点数 (密度代理)
            # [1] 平均速度
            # [2] 速度标准差
            # [3] 局部线性度 (PCA λ_ratio)
            # [4] 平均方向 (归一化)
            # [5] 平均模型概率 (如果有)
            # [6] strip中心x (相对)
            # [7] strip中心y (相对)
            # [8] strip序号 (位置编码代理)
            # [9] strip宽度 (米)

            seq_f = np.zeros(F, dtype=np.float32)
            seq_f[0] = np.log1p(n_pts)
            seq_f[1] = (spd_i.mean() - speed_mean) / max(speed_std, 1e-6)
            seq_f[2] = spd_i.std() / max(speed_std, 1e-6)

            # 局部线性度
            if n_pts >= 5:
                _, _, ev = compute_pca_direction(pts_i)
                if ev[-1] > 1e-10:
                    seq_f[3] = 1.0 - ev[-2] / ev[-1]

            seq_f[4] = dir_i.mean() / 360.0

            if cluster_probs is not None:
                seq_f[5] = cluster_probs[mask].mean()

            seq_f[6] = pts_i[:, 0].mean()
            seq_f[7] = pts_i[:, 1].mean()
            seq_f[8] = (i - n_strips / 2) / (n_strips / 2)  # 归一化到 [-1, 1]
            seq_f[9] = (bins[i + 1] - bins[i])  # strip宽度

            sequence[i] = seq_f

    info = {
        "n_strips": n_strips,
        "n_valid": int(validity.sum()),
        "eigvals": eigvals.tolist(),
        "main_dir": main_dir.tolist(),
        "normal_dir": normal_dir.tolist(),
        "proj_range": [float(proj_min), float(proj_max)],
    }
    return sequence, validity, strip_centers, info


# ═══════════════════════════════════════════════════════════════════
# Part 3: 全量数据 → 条带序列数据集
# ═══════════════════════════════════════════════════════════════════

def build_strip_dataset(
    coords_m: np.ndarray,        # (N, 2) 全部点
    speed: np.ndarray,           # (N,)
    direction: np.ndarray,       # (N,)
    labels: np.ndarray,          # (N,) 0/1 标签
    probs: Optional[np.ndarray] = None,  # (N,) 模型概率
    n_strips: int = 64,
    grid_size: float = 20.0,
    min_cluster_points: int = 50,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], dict]:
    """
    全流程: 密度聚类 → 每个聚类转strip序列 → 构建训练数据集。

    Returns:
        sequences:   list of (n_strips_i, F) arrays
        seq_labels:  list of (n_strips_i,) arrays — strip级标签 (多数投票)
        seq_valid:   list of (n_strips_i,) bool arrays
        stats:       统计信息
    """
    n = len(coords_m)

    # Step 1: 密度聚类
    cluster_ids, density_grid, clus_info, grid_meta = density_cluster(
        coords_m, grid_size=grid_size, min_points=min_cluster_points
    )

    # Step 2: 对非聚类点(道路区域)也建序列 — 用空间分块
    unclustered = cluster_ids < 0
    if unclustered.sum() > min_cluster_points:
        # 对未聚类的道路区域做空间网格分块
        road_x, road_y = coords_m[unclustered, 0], coords_m[unclustered, 1]
        road_grid_size = grid_size * 2
        x_min, x_max = road_x.min(), road_x.max()
        y_min, y_max = road_y.min(), road_y.max()
        nx = max(3, int((x_max - x_min) / road_grid_size))
        ny = max(3, int((y_max - y_min) / road_grid_size))
        rx = np.clip(((road_x - x_min) / road_grid_size).astype(np.int32), 0, nx - 1)
        ry = np.clip(((road_y - y_min) / road_grid_size).astype(np.int32), 0, ny - 1)
        road_block_id = ry * nx + rx
        # 重新编号, 接在聚类ID后面
        offset = cluster_ids.max() + 1
        for bid in np.unique(road_block_id):
            block_mask_global = np.where(unclustered)[0][road_block_id == bid]
            if len(block_mask_global) >= min_cluster_points:
                cluster_ids[block_mask_global] = offset
                offset += 1

    # Step 3: 每个聚类 → strip序列
    sequences, seq_labels, seq_valid = [], [], []
    unique_ids = np.unique(cluster_ids[cluster_ids >= 0])

    for cid in unique_ids:
        c_mask = cluster_ids == cid
        c_pts = coords_m[c_mask]
        c_spd = speed[c_mask]
        c_dir = direction[c_mask]
        c_lbl = labels[c_mask]
        c_prob = probs[c_mask] if probs is not None else None

        if len(c_pts) < n_strips:
            continue

        seq, valid, centers, info = cluster_to_strip_sequence(
            c_pts, c_spd, c_dir, c_prob, n_strips=n_strips
        )

        # Strip标签: 多数投票
        proj = (c_pts - c_pts.mean(axis=0)) @ np.array(info["normal_dir"])
        bins = np.linspace(info["proj_range"][0] - 1e-6, info["proj_range"][1] + 1e-6, info["n_strips"] + 1)
        strip_lbl = np.zeros(info["n_strips"], dtype=np.float32)
        for i in range(info["n_strips"]):
            in_strip = (proj >= bins[i]) & (proj < bins[i + 1])
            if in_strip.sum() >= 2:
                strip_lbl[i] = (c_lbl[in_strip].mean() > 0.5).astype(np.float32)
            else:
                strip_lbl[i] = -1  # 忽略

        sequences.append(seq)
        seq_labels.append(strip_lbl)
        seq_valid.append(valid)

    stats = {
        "n_clusters": len(unique_ids),
        "n_sequences": len(sequences),
        "total_strips": sum(s.shape[0] for s in sequences),
        "grid_meta": grid_meta,
    }
    return sequences, seq_labels, seq_valid, stats


# ═══════════════════════════════════════════════════════════════════
# Part 4: PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════

class StripDataset(Dataset):
    """Strip序列数据集 — 供TCN/BiLSTM训练"""

    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: List[np.ndarray],
        validity: List[np.ndarray],
    ):
        self.samples = []
        for seq, lbl, val in zip(sequences, labels, validity):
            if len(seq) >= 8:
                # 只保留有效标签的strip
                keep = lbl >= 0
                if keep.sum() >= 4:
                    self.samples.append((seq, lbl, val))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, lbl, val = self.samples[idx]
        return (
            torch.from_numpy(seq.copy()),
            torch.from_numpy(lbl.copy()),
            torch.from_numpy(val.copy()),
        )


def collate_strip_batch(batch):
    """变长序列padding + batch"""
    seqs, lbls, vals = zip(*batch)
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)

    B = len(seqs)
    F = seqs[0].shape[1]
    padded_seqs = torch.zeros(B, max_len, F)
    padded_lbls = torch.full((B, max_len), -1.0)  # -1 = ignore
    padded_vals = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (s, l, v) in enumerate(zip(seqs, lbls, vals)):
        L = len(s)
        padded_seqs[i, :L] = s[:L]
        padded_lbls[i, :L] = l[:L]
        padded_vals[i, :L] = v[:L]

    mask = padded_lbls >= 0  # 有效标签mask
    return padded_seqs, padded_lbls, padded_vals, mask, torch.tensor(lengths)
