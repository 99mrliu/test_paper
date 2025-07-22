#!/usr/bin/env python3
# -------------------------------------------------------------
# 读取原始 pairs.tsv → 生成
#   9bit_Dataset_prediction.npz   (float32 or uint8)
#   9bit_Dataset_meta.tsv         (完整 6 列元信息)
# -------------------------------------------------------------
import pandas as pd, numpy as np, os, argparse

# ------------ 7‑bit & 9‑bit 编码函数（保持与训练一致） ------------
pair_to_7bits = {  # 省略其它行，完整映射照旧填全
    ('A','A'):[1,0,0,0,0,0,0], ('A','T'):[1,1,0,0,0,1,0],
    ('A','C'):[1,0,0,1,0,1,0], ('A','G'):[1,0,1,0,0,1,0],
    ('T','T'):[0,1,0,0,0,0,0], ('T','A'):[1,1,0,0,0,0,1],
    ('T','C'):[0,1,0,1,0,1,0], ('T','G'):[0,1,1,0,0,1,0],
    ('G','G'):[0,0,1,0,0,0,0], ('G','A'):[1,0,1,0,0,0,1],
    ('G','T'):[0,1,1,0,0,0,1], ('G','C'):[0,0,1,1,0,1,0],
    ('C','C'):[0,0,0,1,0,0,0], ('C','A'):[1,0,0,1,0,0,1],
    ('C','T'):[0,1,0,1,0,0,1], ('C','G'):[0,0,1,1,0,0,1],
}
default7 = [0]*7
def region_bits(i):
    if 1<=i<=15: return [0,1]
    if 16<=i<=20: return [1,1]
    return [1,0]          # 21‑23

def encode_pair(ot, sg):
    if sg=='N': sg = ot
    return pair_to_7bits.get((ot,sg), default7)

def encode_row(dna, sgrna):
    mat = [encode_pair(ot, sg) + region_bits(i)
           for i,(ot,sg) in enumerate(zip(dna.upper(), sgrna.upper()),1)]
    return np.array(mat, dtype=np.uint8)          # (23,9)

# ---------------- 主流程 ----------------
def build(in_tsv, out_npz='9bit_Dataset_prediction.npz',
          out_meta='9bit_Dataset_meta.tsv', save_dtype='uint8'):
    df = pd.read_csv(in_tsv, sep=r'\s+', header=0, dtype=str)   # 有表头
    if df.shape[1] < 4:
        raise ValueError('输入至少 4 列 (sgRNA / DNA 在第1/4列)')
    sg  = df.iloc[:,0].str.upper()
    dna = df.iloc[:,3].str.upper()
    ok  = (sg.str.len()==23) & (dna.str.len()==23)
    if not ok.all():
        print(f'⚠ 有 {(~ok).sum()} 条长度≠23 被忽略')
        df, sg, dna = df[ok], sg[ok], dna[ok]

    mats = np.stack([encode_row(d,dg) for d,dg in zip(dna, sg)])
    np.savez_compressed(out_npz, data=mats.astype(save_dtype))
    df.to_csv(out_meta, sep='\t', index=False)
    print(f'✓ data→ {out_npz}  ({mats.shape}, dtype={save_dtype})')
    print(f'✓ meta→ {out_meta}  ({df.shape[0]} 条)')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='tsv', required=True, help='原始 pairs.tsv')
    ap.add_argument('--dtype', choices=['uint8','float32'], default='uint8')
    args = ap.parse_args()
    build(args.tsv,
          out_npz='9bit_Dataset_prediction.npz',
          out_meta='9bit_Dataset_meta.tsv',
          save_dtype=args.dtype)
