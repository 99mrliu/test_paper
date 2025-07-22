#!/usr/bin/env python3
"""
Hard‑coded prediction script  · 2025‑05‑10
运行: python predict_offtarget.py
"""

# ========= 配置区 =========
MODEL_PATH   = "models/best_model_siteseq.pth"
NPZ_PATH     = "9bit_Dataset_prediction.npz"
META_PATH    = "9bit_Dataset_meta.tsv"
USE_CPU_ONLY = True          # True→CPU, False→GPU(如可用)
TOP_K        = 20
# ==========================

import numpy as np
import pandas as pd
import torch
from model_def import CustomModel   # 你的模型结构

# ---------- 工具函数 ----------
def load_npz_float32(path):
    arr = np.load(path)["data"]
    return arr.astype(np.float32) if arr.dtype != np.float32 else arr

def batch_predict(model, data, device, bs=1024):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(data), bs):
            x = torch.tensor(data[i:i+bs], device=device)
            outs.append(model(x).squeeze().cpu().numpy())
    return np.concatenate(outs)

# ---------- 主流程 ----------
def main():
    device = torch.device("cpu") if USE_CPU_ONLY or not torch.cuda.is_available() \
             else torch.device("cuda")
    print(">>> device:", device)

    # 1) 读数据 & 元信息
    data = load_npz_float32(NPZ_PATH)
    meta = pd.read_csv(META_PATH, sep="\t")
    if len(meta) != len(data):
        raise ValueError(f"行数不一致: npz={len(data)}, meta={len(meta)}")

    # 2) 加载模型
    model = CustomModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(">>> model loaded")

    # 3) 推理
    probs = batch_predict(model, data, device)
    meta["prob_offtarget"] = probs
    meta.sort_values("prob_offtarget", ascending=False, inplace=True)

    # 4) 保存
    out_file = "prediction_with_prob_siteseq.tsv"
    meta.to_csv(out_file, sep="\t", index=False)
    print(f"✓ 预测完成，已保存 {out_file}")

    # 5) 终端打印 TOP_K，自动识别列并重命名
    sg_col  = "sgRNA" if "sgRNA" in meta.columns else meta.columns[0]
    dna_col = "DNA"   if "DNA"   in meta.columns else meta.columns[3]

    display_df = meta[[sg_col, dna_col, "prob_offtarget"]].head(TOP_K).copy()
    display_df.columns = ["sgRNA", "DNA", "prob_offtarget"]   # 统一标题

    print(f"\n=== TOP {TOP_K} ===")
    print(display_df.to_string(index=False))

# ---------- 入口 ----------
if __name__ == "__main__":
    main()
