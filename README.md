<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License">
  <img src="https://img.shields.io/badge/Python-3.8+-yellow" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="PyTorch">
</p>

<h1 align="center">🚜 FRTMiner</h1>
<h3 align="center"><em>Field-Road Trajectory Miner</em></h3>

<p align="center">
  <b>Efficient Farmland Parcel Delineation and Field Road Network Extraction<br/>from Pure GPS Trajectories — No Imagery Required.</b>
</p>

<br/>

## 📖 Overview

**FRTMiner** is a cascaded hybrid model designed to solve a fundamental challenge in precision agriculture:

> *How can we identify farmland parcels and field road networks efficiently and at low cost, using only agricultural machinery GPS data — without any remote sensing imagery?*

Traditional image-based segmentation methods rely on expensive high-resolution imagery and heavy computation. Existing trajectory-based methods often treat parcels and roads as separate entities, overlooking a key topological insight: **field roads are fundamentally the edges of farmland.** FRTMiner bridges this gap with a novel "coarse-to-fine" architecture that mimics how a human expert thinks — first locating the fields, then tracing the roads along their boundaries.

<br/>

## 🎯 Core Problem & Motivation

| Challenge | Traditional Approach | FRTMiner's Innovation |
|:---|:---|:---|
| **Data Dependency** | Requires high-resolution satellite/UAV imagery | Works with low-cost GPS logs only |
| **Computational Cost** | GPU-intensive image segmentation | Lightweight sequence modeling |
| **Contextual Confusion** | Struggles to distinguish roads from field edges | Explicitly models the parcel-road topology |
| **Pattern Ambiguity** | Cannot separate transfer paths from headland turns | Sequence-context classifier captures motion dynamics |

<br/>

## 🧠 Core Architecture

FRTMiner employs a **two-stage, coarse-to-fine, long-chain reasoning pipeline**:

```mermaid
flowchart LR
    A[🛰️ Raw GPS<br/>Trajectories] --> B[🧩 Stage I<br/>Field Coarse<br/>Segmentation]
    B --> C[📐 Field Boundary<br/>Polygons]
    C --> D[🔍 Stage II<br/>Edge Refinement]
    A --> D
    D --> E[🛤️ Field Road<br/>Network]

    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style E fill:#fce4ec
