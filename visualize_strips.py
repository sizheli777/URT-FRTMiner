"""
Strip序列化 + 二分类可视化

展示:
  1. 密度聚类结果 (高密田块 vs 低密道路)
  2. 单个聚类的 strip 切分 (PCA方向 + strip条带)
  3. Strip特征序列的热力图
  4. 二分类预测结果 (田/路 沿法线方向的过渡)
"""

import os
import sys
import argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Fix CJK font — try SimHei or fall back to sans-serif
try:
    matplotlib.font_manager.findfont("SimHei", fallback_to_default=False)
    plt.rcParams["font.sans-serif"] = ["SimHei"] + plt.rcParams["font.sans-serif"]
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig
from model import (
    compute_derived_features,
    compute_temporal_features,
    load_and_preprocess_csv,
    TrajectoryFormer,
)
from inference import build_features, sliding_inference
from strip_sequence import (
    density_cluster,
    cluster_to_strip_sequence,
    build_strip_dataset,
)
from model.edge_refiner import StripRefiner, MaskedFocalLoss


def latlon_to_meters(lat, lon, ref_lat, ref_lon):
    R = 6378137.0
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    ref_lat_r = np.radians(ref_lat)
    x = R * (lon_r - np.radians(ref_lon)) * np.cos(ref_lat_r)
    y = R * (lat_r - ref_lat_r)
    return x, y


def visualize_clusters_and_strips(
    coords_m, cluster_labels, density_grid, grid_meta,
    lat, lon, speed, labels,
    save_dir="strip_viz",
):
    """
    图1: 密度聚类概览
      - 左上: 密度网格热力图
      - 右上: 聚类结果散点图 (颜色=聚类ID)
      - 下: 聚类统计
    """
    os.makedirs(save_dir, exist_ok=True)
    x, y = coords_m[:, 0], coords_m[:, 1]
    x_min, x_max, y_min, y_max = grid_meta

    fig, axes = plt.subplots(1, 3, figsize=(28, 9))
    fig.suptitle("Stage 1: Density Grid Clustering — Field Block Discovery", fontsize=14, fontweight="bold")

    # --- 左: 密度网格 ---
    ax = axes[0]
    im = ax.imshow(density_grid, origin="lower", cmap="YlOrRd",
                   extent=[x_min, x_max, y_min, y_max], aspect="auto")
    ax.set_title("Density Grid (20m cells)")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Points per cell")

    # --- 中: 聚类结果 ---
    ax = axes[1]
    unique_ids = np.unique(cluster_labels)
    cmap = plt.cm.tab20
    for cid in unique_ids:
        if cid < 0:
            mask = cluster_labels == cid
            ax.scatter(x[mask], y[mask], c="#888888", s=0.3, alpha=0.3, rasterized=True, label="Low density")
        else:
            mask = cluster_labels == cid
            ax.scatter(x[mask], y[mask], c=[cmap(cid % 20)], s=1.0, alpha=0.7, rasterized=True)
    ax.set_title(f"Density Clusters ({len(unique_ids[unique_ids>=0])} high-density regions)")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal")
    # deduplicate legend
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), markerscale=6, fontsize=7)

    # --- 右: 聚类统计 ---
    ax = axes[2]
    ax.axis("off")
    n_clustered = (cluster_labels >= 0).sum()
    n_total = len(cluster_labels)
    stats = (
        f"Cluster Statistics\n"
        f"{'='*30}\n\n"
        f"Total points:       {n_total:>8d}\n"
        f"Clustered (field):  {n_clustered:>8d} ({n_clustered/n_total:.1%})\n"
        f"Low-density (road): {n_total-n_clustered:>8d} ({(n_total-n_clustered)/n_total:.1%})\n\n"
        f"Num clusters:       {len(unique_ids[unique_ids>=0]):>8d}\n"
        f"Grid shape:         {density_grid.shape}\n"
    )
    ax.text(0.1, 0.95, stats, transform=ax.transAxes, fontsize=12,
            fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(save_dir, "01_density_clusters.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [保存] {path}")


def visualize_single_cluster_strips(
    coords_m, cluster_labels, speed, direction, labels,
    probs=None, n_strips=64, max_clusters=4,
    save_dir="strip_viz",
):
    """
    图2: 选取几个聚类, 详细展示 strip 切分过程。
    每个聚类一行三列:
      - 左: 聚类点云 + PCA方向箭头 + strip条带
      - 中: Strip特征热力图 (n_strips × F)
      - 右: 沿法线方向的标签过渡 (strip 级 + 点级)
    """
    unique_ids = np.unique(cluster_labels[cluster_labels >= 0])
    if len(unique_ids) == 0:
        print("  无聚类可展示")
        return

    # 选最大的几个聚类
    cluster_sizes = [(cid, (cluster_labels == cid).sum()) for cid in unique_ids]
    cluster_sizes.sort(key=lambda x: -x[1])
    selected = [cid for cid, _ in cluster_sizes[:max_clusters]]

    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 3, figsize=(24, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Stage 2: Strip-based Sequence Ordering — Per-Cluster Detail", fontsize=14, fontweight="bold")

    for row_idx, cid in enumerate(selected):
        c_mask = cluster_labels == cid
        c_pts = coords_m[c_mask]
        c_spd = speed[c_mask]
        c_dir = direction[c_mask]
        c_lbl = labels[c_mask]
        c_prob = probs[c_mask] if probs is not None else None

        # Strip切分
        seq, valid, centers, info = cluster_to_strip_sequence(
            c_pts, c_spd, c_dir, c_prob, n_strips=n_strips
        )

        # --- 左: 聚类点云 + strip ---
        ax = axes[row_idx, 0]
        mean_pt = c_pts.mean(axis=0)
        main_dir = np.array(info["main_dir"])
        normal_dir = np.array(info["normal_dir"])

        # 背景: 所有点
        ax.scatter(c_pts[:, 0], c_pts[:, 1], c="#cccccc", s=0.5, alpha=0.4, rasterized=True)

        # 按strip着色
        proj = (c_pts - mean_pt) @ normal_dir
        bins = np.linspace(np.array(info["proj_range"])[0] - 1e-6,
                           np.array(info["proj_range"])[1] + 1e-6,
                           info["n_strips"] + 1)
        for i in range(info["n_strips"]):
            in_s = (proj >= bins[i]) & (proj < bins[i + 1])
            if in_s.sum() > 0:
                color = "#ff5533" if (c_lbl[in_s].mean() > 0.5) else "#3388ff"
                ax.scatter(c_pts[in_s, 0], c_pts[in_s, 1], c=color, s=1.5, alpha=0.7, rasterized=True)

        # PCA方向箭头
        scale = 30.0
        ax.arrow(mean_pt[0], mean_pt[1],
                 main_dir[0] * scale, main_dir[1] * scale,
                 head_width=8, head_length=12, fc="red", ec="red", alpha=0.8, label="Main (沿路)")
        ax.arrow(mean_pt[0], mean_pt[1],
                 normal_dir[0] * scale, normal_dir[1] * scale,
                 head_width=8, head_length=12, fc="green", ec="green", alpha=0.8, label="Normal (切片)")

        # strip 边界线
        for b in bins[1:-1]:
            p0 = mean_pt + main_dir * (-scale * 2) + normal_dir * b
            p1 = mean_pt + main_dir * (scale * 2) + normal_dir * b
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "k-", linewidth=0.3, alpha=0.3)

        ax.set_title(f"Cluster {cid} ({len(c_pts)} pts) — λ=[{info['eigvals'][0]:.0f}, {info['eigvals'][1]:.0f}]")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_aspect("equal")
        ax.legend(fontsize=7)

        # --- 中: 特征热力图 ---
        ax = axes[row_idx, 1]
        valid_seq = seq  # seq shape (n_strips, F)
        im = ax.imshow(valid_seq.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
        ax.set_xlabel("Strip index")
        ax.set_ylabel("Feature dim")
        feature_names = ["log_dens", "avg_spd", "std_spd", "linearity", "avg_dir", "prob", "cx", "cy", "pos", "width"]
        ax.set_yticks(range(min(10, valid_seq.shape[1])))
        ax.set_yticklabels(feature_names[:valid_seq.shape[1]], fontsize=7)
        ax.set_title("Strip Feature Heatmap")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # --- 右: 标签过渡 ---
        ax = axes[row_idx, 2]
        strip_indices = np.arange(info["n_strips"])

        # Strip级标签
        strip_lbl = np.zeros(info["n_strips"])
        for i in range(info["n_strips"]):
            in_s = (proj >= bins[i]) & (proj < bins[i + 1])
            if in_s.sum() > 0:
                strip_lbl[i] = c_lbl[in_s].mean()

        ax.fill_between(strip_indices, 0, strip_lbl, alpha=0.3, color="orange", label="Strip field ratio")
        ax.plot(strip_indices, strip_lbl, "o-", color="#ff5533", markersize=3, linewidth=1, label="Field ratio")

        # 点级标签 (聚合到strip的中心位置)
        proj_centers = np.linspace(info["proj_range"][0], info["proj_range"][1], info["n_strips"])
        ax2 = ax.twinx()
        ax2.bar(strip_indices,
                [((proj >= bins[i]) & (proj < bins[i+1])).sum() for i in range(info["n_strips"])],
                alpha=0.3, color="gray", width=0.8)
        ax2.set_ylabel("Points per strip", fontsize=8)

        ax.axhline(y=0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Strip index (along normal direction)")
        ax.set_ylabel("Field ratio")
        ax.set_title(f"Boundary Transition — {info['n_valid']}/{info['n_strips']} valid strips")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    path = os.path.join(save_dir, "02_strip_sequences.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [保存] {path}")


def visualize_model_prediction(
    coords_m, cluster_labels, speed, direction, labels,
    model=None, device=None, n_strips=64,
    save_dir="strip_viz",
):
    """
    图3: 使用 StripRefiner 做二分类预测, 对比真实标签。

    如果在没有训练好的模型的情况下运行, 使用启发式规则模拟。
    """
    sequences, seq_labels, seq_valid, stats = build_strip_dataset(
        coords_m, speed, direction, labels, n_strips=n_strips
    )

    if len(sequences) == 0:
        print("  无有效序列")
        return

    # 使用模型或启发式规则
    all_preds = []
    if model is not None and device is not None:
        model.eval()
        for seq in sequences:
            x = torch.from_numpy(seq).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits).squeeze(0).squeeze(-1).cpu().numpy()
            all_preds.append((probs > 0.5).astype(np.int32))
    else:
        # 启发式: 基于速度和密度的简易预测
        for seq in sequences:
            # seq[:,1]=avg_speed, seq[:,3]=linearity
            # 道路: 高速+高线性度; 田: 低速+低线性度
            road_score = seq[:, 1] * 0.3 + seq[:, 2] * 0.2 - seq[:, 0] * 0.1 + seq[:, 3] * 0.4
            pred = (road_score > 0.0).astype(np.int32)
            all_preds.append(pred)

    # 对比可视化: 选几个序列展示
    n_show = min(4, len(all_preds))
    fig, axes = plt.subplots(n_show, 2, figsize=(20, 5 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Stage 3: Strip Classification — Predicted vs Ground Truth", fontsize=14, fontweight="bold")

    for i in range(n_show):
        seq = sequences[i]
        true_lbl = seq_labels[i]
        pred_lbl = all_preds[i]

        strip_idx = np.arange(len(true_lbl))
        valid = true_lbl >= 0

        # --- 左: 真实标签 ---
        ax = axes[i, 0]
        colors = ["#ff5533" if tl > 0.5 else "#3388ff" if tl >= 0 else "#888888"
                  for tl in true_lbl]
        ax.bar(strip_idx, np.ones(len(true_lbl)), color=colors, width=1.0, edgecolor="none")
        ax.set_ylabel("Strip")
        ax.set_xlabel("Strip index")
        ax.set_title(f"Sequence {i+1} — Ground Truth")
        ax.set_ylim(0, 1)
        ax.set_yticks([])

        legend_elements = [
            mpatches.Patch(color="#ff5533", label="Field (1)"),
            mpatches.Patch(color="#3388ff", label="Road (0)"),
            mpatches.Patch(color="#888888", label="Ignore (-1)"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc="upper right")

        # --- 右: 预测 ---
        ax = axes[i, 1]
        colors = ["#ff5533" if p > 0.5 else "#3388ff" for p in pred_lbl]
        ax.bar(strip_idx, np.ones(len(pred_lbl)), color=colors, width=1.0, edgecolor="none")
        ax.set_xlabel("Strip index")
        ax.set_title(f"Sequence {i+1} — Prediction (heuristic)")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.legend(handles=legend_elements[:2], fontsize=7, loc="upper right")

        # 准确率标注
        if valid.sum() > 0:
            acc = (pred_lbl[valid] == true_lbl[valid]).mean()
            axes[i, 0].text(0.02, 0.9, f"Acc: {acc:.1%}", transform=axes[i, 0].transAxes,
                           fontsize=10, fontfamily="monospace")

    plt.tight_layout()
    path = os.path.join(save_dir, "03_strip_classification.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [保存] {path}")


def visualize_full_pipeline(
    coords_m, cluster_labels, speed, direction, labels,
    predicted_labels=None,
    save_dir="strip_viz",
):
    """
    图4: 全流程对比 — 原始点云 vs 密度聚类 vs Strip分类 vs 最终分割
    """
    fig, axes = plt.subplots(2, 2, figsize=(22, 20))
    fig.suptitle("Full Pipeline: Raw → Clusters → Strips → Segmentation", fontsize=14, fontweight="bold")

    x, y = coords_m[:, 0], coords_m[:, 1]
    road_c = "#3388ff"
    field_c = "#ff5533"

    # --- 左上: 原始点云 + 真实标签 ---
    ax = axes[0, 0]
    for lbl, color, name in [(0, road_c, "Road"), (1, field_c, "Field")]:
        mask = labels == lbl
        ax.scatter(x[mask], y[mask], c=color, s=0.5, alpha=0.6, rasterized=True, label=name)
    ax.set_title("Original Points (Ground Truth)")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal")
    ax.legend(markerscale=8)

    # --- 右上: 密度聚类 ---
    ax = axes[0, 1]
    unique_ids = np.unique(cluster_labels)
    cmap = plt.cm.tab20
    for cid in unique_ids:
        mask = cluster_labels == cid
        if cid < 0:
            ax.scatter(x[mask], y[mask], c="#888888", s=0.3, alpha=0.2, rasterized=True)
        else:
            ax.scatter(x[mask], y[mask], c=[cmap(cid % 20)], s=1.0, alpha=0.6, rasterized=True)
    ax.set_title(f"Density Clustering ({len(unique_ids[unique_ids>=0])} field blocks)")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal")

    # --- 左下: Strip预测 (如果有) ---
    ax = axes[1, 0]
    if predicted_labels is not None:
        for lbl, color, name in [(0, road_c, "Road"), (1, field_c, "Field")]:
            mask = predicted_labels == lbl
            ax.scatter(x[mask], y[mask], c=color, s=0.5, alpha=0.6, rasterized=True, label=name)
        ax.set_title("Strip Refiner Prediction")
        ax.legend(markerscale=8)
    else:
        ax.scatter(x, y, c="#cccccc", s=0.5, alpha=0.3, rasterized=True)
        ax.set_title("Strip Refiner Prediction (no model loaded)")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal")

    # --- 右下: 统计对比 ---
    ax = axes[1, 1]
    ax.axis("off")
    n_total = len(labels)
    n_clustered = (cluster_labels >= 0).sum()

    stats_lines = [
        "Pipeline Statistics",
        "=" * 35,
        "",
        f"Total points:            {n_total:>8d}",
        f"Clustered (high density): {n_clustered:>8d} ({n_clustered/n_total:.1%})",
        f"Road (label=0):           {(labels==0).sum():>8d} ({(labels==0).sum()/n_total:.1%})",
        f"Field (label=1):          {(labels==1).sum():>8d} ({(labels==1).sum()/n_total:.1%})",
        "",
        "Architecture:",
        "  Stage 1: Density Grid Clustering",
        "  Stage 2: PCA + Strip Ordering",
        "  Stage 3: TCN / BiLSTM-Attn Classifier",
        "",
        "Strip features: [density, speed, linearity, ...]",
        "Sequence: equal-width bins along PCA normal direction",
    ]
    if predicted_labels is not None:
        acc = (predicted_labels == labels).mean()
        stats_lines.extend([
            "",
            f"Overall Accuracy: {acc:.1%}",
        ])

    ax.text(0.05, 0.95, "\n".join(stats_lines), transform=ax.transAxes,
            fontsize=10, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(save_dir, "04_full_pipeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [保存] {path}")


def main():
    parser = argparse.ArgumentParser(description="Strip序列化 + 二分类可视化")
    parser.add_argument("--data", type=str, required=True, help="CSV/XLS数据路径")
    parser.add_argument("--output", type=str, default="strip_viz", help="输出目录")
    parser.add_argument("--n_strips", type=int, default=64, help="每个聚类的strip数")
    parser.add_argument("--grid_size", type=float, default=20.0, help="密度网格大小(米)")
    parser.add_argument("--max_clusters", type=int, default=4, help="展示的聚类数")
    parser.add_argument("--checkpoint", type=str, default=None, help="TrajectoryFormer模型(可选)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ── 加载数据 ──
    print("加载数据...")
    import pandas as pd
    ext = os.path.splitext(args.data)[1].lower()
    if ext in (".xls", ".xlsx"):
        from model.dataset import load_single_file
        lat, lon, speed, direction, altitude, time_sec, labels = load_single_file(args.data)
    else:
        df = pd.read_csv(args.data)
        # 按列名读取
        col_map = {c.lower(): c for c in df.columns}
        lat = df[col_map.get("latitude", col_map.get("lat", df.columns[0]))].values.astype(np.float32)
        lon = df[col_map.get("longitude", col_map.get("lon", df.columns[1]))].values.astype(np.float32)
        time_sec = df[col_map.get("time", df.columns[2])].values.astype(np.float64)
        speed = df[col_map.get("speed", df.columns[3])].values.astype(np.float32)
        direction = df[col_map.get("direction", df.columns[4])].values.astype(np.float32)
        altitude = df.get(col_map.get("altitude", ""), pd.Series(np.zeros(len(df)))).values.astype(np.float32)
        labels = df[col_map.get("label", df.columns[6])].values.astype(np.int64)
        speed = np.clip(speed, 0, 50.0)
        direction = np.fmod(direction, 360.0)

    # 转换为米制坐标
    ref_lat, ref_lon = lat.mean(), lon.mean()
    coords_m = np.column_stack(latlon_to_meters(lat, lon, ref_lat, ref_lon))
    print(f"总点数: {len(lat)}, 道路={int((labels==0).sum())}, 田间={int((labels==1).sum())}")

    # ── 可选: 加载模型获取概率 ──
    probs = None
    if args.checkpoint:
        print("加载TrajectoryFormer模型...")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = checkpoint.get("config", ModelConfig())
        model = TrajectoryFormer(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        features = build_features(lat, lon, speed, direction, altitude, time_sec)
        mean, std = features.mean(axis=0, keepdims=True), features.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        features = (features - mean) / std
        probs = sliding_inference(model, features, config, device)
        print(f"  模型概率范围: [{probs.min():.3f}, {probs.max():.3f}]")

    # ── Stage 1: 密度聚类 ──
    print("\n=== Stage 1: 密度聚类 ===")
    cluster_labels, density_grid, clus_info, grid_meta = density_cluster(
        coords_m, grid_size=args.grid_size
    )
    print(f"  聚类数: {clus_info['num_regions']}, 参与聚类点数: {clus_info['num_clustered']}")

    # ── 可视化 ──
    print("\n生成可视化...")
    os.makedirs(args.output, exist_ok=True)

    visualize_clusters_and_strips(
        coords_m, cluster_labels, density_grid, grid_meta,
        lat, lon, speed, labels,
        save_dir=args.output,
    )

    visualize_single_cluster_strips(
        coords_m, cluster_labels, speed, direction, labels,
        probs=probs, n_strips=args.n_strips, max_clusters=args.max_clusters,
        save_dir=args.output,
    )

    visualize_model_prediction(
        coords_m, cluster_labels, speed, direction, labels,
        model=None, device=None, n_strips=args.n_strips,
        save_dir=args.output,
    )

    visualize_full_pipeline(
        coords_m, cluster_labels, speed, direction, labels,
        save_dir=args.output,
    )

    print(f"\n所有图片已保存到: {args.output}/")


if __name__ == "__main__":
    main()
