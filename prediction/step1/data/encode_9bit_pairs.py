import pandas as pd
import numpy as np
import os

# -------------------------------------------------------------
# 1. 7‑bit 碱基对编码映射
# -------------------------------------------------------------
pair_to_7bits = {
    ('A','A'):[1,0,0,0,0,0,0], ('A','T'):[1,1,0,0,0,1,0],
    ('A','C'):[1,0,0,1,0,1,0], ('A','G'):[1,0,1,0,0,1,0],
    ('T','T'):[0,1,0,0,0,0,0], ('T','A'):[1,1,0,0,0,0,1],
    ('T','C'):[0,1,0,1,0,1,0], ('T','G'):[0,1,1,0,0,1,0],
    ('G','G'):[0,0,1,0,0,0,0], ('G','A'):[1,0,1,0,0,0,1],
    ('G','T'):[0,1,1,0,0,0,1], ('G','C'):[0,0,1,1,0,1,0],
    ('C','C'):[0,0,0,1,0,0,0], ('C','A'):[1,0,0,1,0,0,1],
    ('C','T'):[0,1,0,1,0,0,1], ('C','G'):[0,0,1,1,0,0,1],
}
default_7bits = [0,0,0,0,0,0,0]

def encode_pair7(b1:str, b2:str):
    """返回 7‑bit 编码（大写碱基对）。"""
    return pair_to_7bits.get((b1,b2), default_7bits)

# -------------------------------------------------------------
# 2. 位置 2‑bit 区域位
# -------------------------------------------------------------
def region_bits(pos:int):
    if   1 <= pos <= 15: return [0,1]   # 普通
    elif 16<= pos <= 20: return [1,1]   # Seed
    elif 21<= pos <= 23: return [1,0]   # PAM
    else: raise ValueError('position must be 1‑23')

# -------------------------------------------------------------
# 3. 单条序列对 → (23,9)
# -------------------------------------------------------------
def encode_9bit(offtarget:str, sgrna:str)->np.ndarray:
    matrix = []
    for i,(ot,sg) in enumerate(zip(offtarget.upper(), sgrna.upper()), start=1):
        if sg == 'N': sg = ot   # N → 用 DNA 位替代
        bits9 = encode_pair7(ot, sg) + region_bits(i)
        matrix.append(bits9)
    return np.array(matrix, dtype=np.float32)  # (23,9)

# -------------------------------------------------------------
# 4. 处理文件
# -------------------------------------------------------------
def process_txt(txt_path:str):
    # 只有一行表头？没有就 header=None
    df = pd.read_csv(txt_path, sep=r'\s+', header=None, dtype=str)
    if df.shape[1] < 4:
        raise ValueError('文件至少要有 4 列 (sgRNA 在第1列, DNA 在第4列)')
    sgrnas = df.iloc[:,0].str.upper()
    dnas   = df.iloc[:,3].str.upper()

    lens_ok = (sgrnas.str.len()==23) & (dnas.str.len()==23)
    if not lens_ok.all():
        print(f'⚠ 有 {(~lens_ok).sum()} 条长度≠23 已自动忽略')
        sgrnas, dnas = sgrnas[lens_ok], dnas[lens_ok]

    encoded = np.stack([encode_9bit(dna, sg) for dna,sg in zip(dnas, sgrnas)])
    meta    = df[lens_ok].reset_index(drop=True)   # 方便后续对应概率
    return encoded, meta

# -------------------------------------------------------------
# 5. 主函数
# -------------------------------------------------------------
def main():
    in_txt  = 'processed_out.txt'                  # 修改为你的文件
    out_npz = '9bit_Dataset_prediction.npz'
    os.makedirs(os.path.dirname(out_npz) or '.', exist_ok=True)

    data, meta = process_txt(in_txt)
    np.savez_compressed(out_npz, data=data.astype(np.float32))
    meta.to_csv(out_npz.replace('.npz','_meta.tsv'), sep='\t', index=False)

    print(f'✓ 编码完成: {data.shape}  已保存 {out_npz}')
    print(f'✓ 元信息保存为: {out_npz.replace(".npz","_meta.tsv")}')

if __name__ == '__main__':
    main()
