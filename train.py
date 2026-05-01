"""
训练脚本 — TrajectoryFormer 田路分割模型
支持 .xls / .csv 文件，支持单文件或目录批量加载
"""

import os
import sys
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig
from model import (
    compute_derived_features,
    compute_temporal_features,
    TrajectoryDataset,
    load_multiple_files,
    discover_files,
    load_single_file,
    TrajectoryFormer,
    CombinedLoss,
)


def build_sequences(lat, lon, speed, direction, altitude, time_sec, labels):
    """构建完整特征矩阵"""
    manual_feat = compute_derived_features(lat, lon, speed, direction, time_sec, altitude)
    temporal_feat = compute_temporal_features(time_sec)

    # 相对时间
    t_rel = time_sec - time_sec[0]

    raw_feat = np.stack(
        [lat, lon, speed, direction / 360.0, altitude, t_rel],
        axis=-1,
    )

    all_feat = np.concatenate([raw_feat, manual_feat, temporal_feat], axis=-1)
    return all_feat.astype(np.float32), labels.astype(np.int64)


def normalize_features(train_feat, val_feat, test_feat):
    """基于训练统计做Z-score标准化"""
    mean = train_feat.mean(axis=0, keepdims=True)
    std = train_feat.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (
        (train_feat - mean) / std,
        (val_feat - mean) / std,
        (test_feat - mean) / std,
    )


def compute_metrics(preds, labels):
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).squeeze(-1)
        preds = (probs > 0.5).long().cpu().numpy()
        all_preds.append(preds.ravel())
        all_labels.append(y.cpu().numpy().ravel())

    if not all_preds:
        return {"loss": float("inf"), "accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_labels))
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def train(config, train_loader, val_loader, device):
    model = TrajectoryFormer(config).to(device)
    # alpha=0.1 → road weight=0.9, field weight=0.1 (强偏向道路类)
    criterion = CombinedLoss(focal_weight=0.6, dice_weight=0.4, alpha=0.1, gamma=2.0).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    best_f1 = 0.0
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(config.max_epochs):
        model.train()
        train_loss = 0.0
        all_preds, all_labels = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.max_epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs > 0.5).long().detach().cpu().numpy()
            all_preds.append(preds.ravel())
            all_labels.append(y.detach().cpu().numpy().ravel())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_labels))
        train_metrics["loss"] = train_loss / max(len(train_loader), 1)
        history["train"].append(train_metrics)

        val_metrics = evaluate(model, val_loader, criterion, device)
        history["val"].append(val_metrics)

        print(f"  Train | loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} "
              f"f1={train_metrics['f1']:.4f}")
        print(f"  Val   | loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
              f"f1={val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "config": config,
                 "best_f1": best_f1},
                os.path.join(config.checkpoint_dir, "best_model.pt"),
            )
            print(f"  → 保存最佳模型 (F1={best_f1:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= config.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\n训练完成。最佳F1={best_f1:.4f}")
    with open(os.path.join(config.checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2, default=float)
    return model


def main():
    parser = argparse.ArgumentParser(description="TrajectoryFormer 田路分割训练")
    parser.add_argument("--data", type=str, required=True,
                        help="CSV/XLS数据路径 或 目录")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = ModelConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_len=args.seq_len,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据 — 支持单文件或目录
    print("加载数据...")
    if os.path.isdir(args.data):
        files = discover_files(args.data)
        lat, lon, speed, direction, altitude, time_sec, labels = load_multiple_files(files)
    elif os.path.isfile(args.data):
        lat, lon, speed, direction, altitude, time_sec, labels = load_single_file(args.data)
    else:
        raise FileNotFoundError(f"找不到: {args.data}")

    print(f"总点数: {len(lat)}")
    print(f"  道路(0): {(labels==0).sum()} ({(labels==0).sum()/len(labels):.1%})")
    print(f"  田间(1): {(labels==1).sum()} ({(labels==1).sum()/len(labels):.1%})")

    # 构建特征
    print("构建特征...")
    features, labels = build_sequences(lat, lon, speed, direction, altitude, time_sec, labels)
    print(f"特征维度: {features.shape[1]}")

    # 分割数据
    # 随机打乱索引再分割 (避免按文件顺序导致的分布偏差)
    n = len(features)
    indices = np.random.permutation(n)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_feat, train_lbl = features[train_idx], labels[train_idx]
    val_feat, val_lbl = features[val_idx], labels[val_idx]
    test_feat, test_lbl = features[test_idx], labels[test_idx]

    # 标准化
    train_feat, val_feat, test_feat = normalize_features(train_feat, val_feat, test_feat)

    # 创建数据集
    train_ds = TrajectoryDataset(train_feat, train_lbl, config.max_seq_len, config.seq_overlap)
    val_ds = TrajectoryDataset(val_feat, val_lbl, config.max_seq_len, config.seq_overlap)
    test_ds = TrajectoryDataset(test_feat, test_lbl, config.max_seq_len, config.seq_overlap)

    print(f"序列数 — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # 计算每个窗口的采样权重 (道路占比越高权重越大)
    train_weights = []
    for i in range(len(train_ds)):
        _, y = train_ds[i]
        road_ratio = (y == 0).float().mean().item()
        # 道路占比越高, 权重越大 (最多5倍)
        weight = 1.0 + road_ratio * 8.0
        train_weights.append(weight)

    sampler = WeightedRandomSampler(
        torch.tensor(train_weights), num_samples=len(train_ds) * 2, replacement=True
    )

    use_pin = device.type == "cuda"
    num_workers = 2 if device.type == "cuda" else 0

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=use_pin)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=use_pin)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=use_pin)

    # 训练
    print("开始训练...")
    model = train(config, train_loader, val_loader, device)

    # 测试
    print("\n评估测试集...")
    criterion = CombinedLoss(focal_weight=0.6, dice_weight=0.4, alpha=0.1, gamma=2.0)
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test | acc={test_metrics['accuracy']:.4f} prec={test_metrics['precision']:.4f} "
          f"rec={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
