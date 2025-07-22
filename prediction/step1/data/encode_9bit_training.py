import pandas as pd
import numpy as np

# =====================================================================
# 1. 定义 7-bit 编码映射
# =====================================================================

pair_to_7bits = {
    # 完全匹配（方向 bits 均为 0,0）
    ('A', 'A'): [1, 0, 0, 0, 0, 0, 0],  # A + A
    ('A', 'T'): [1, 1, 0, 0, 0, 1, 0],  # A + T
    ('A', 'C'): [1, 0, 0, 1, 0, 1, 0],  # A + C
    ('A', 'G'): [1, 0, 1, 0, 0, 1, 0],  # A + G

    # T 开头的碱基对
    ('T', 'T'): [0, 1, 0, 0, 0, 0, 0],  # T + T
    ('T', 'A'): [1, 1, 0, 0, 0, 0, 1],  # T + A
    ('T', 'C'): [0, 1, 0, 1, 0, 1, 0],  # T + C
    ('T', 'G'): [0, 1, 1, 0, 0, 1, 0],  # T + G

    # G 开头的碱基对
    ('G', 'G'): [0, 0, 1, 0, 0, 0, 0],  # G + G
    ('G', 'A'): [1, 0, 1, 0, 0, 0, 1],  # G + A
    ('G', 'T'): [0, 1, 1, 0, 0, 0, 1],  # G + T
    ('G', 'C'): [0, 0, 1, 1, 0, 1, 0],  # G + C

    # C 开头的碱基对
    ('C', 'C'): [0, 0, 0, 1, 0, 0, 0],  # C + C
    ('C', 'A'): [1, 0, 0, 1, 0, 0, 1],  # C + A
    ('C', 'T'): [0, 1, 0, 1, 0, 0, 1],  # C + T
    ('C', 'G'): [0, 0, 1, 1, 0, 0, 1],  # C + G

    # 其它示例：A-A, G-G 已在上方完全匹配示例
    # 如果有更多组合，如 (A,G), (G,A), (A,C), (C,A) 等，需要您根据需求补充
}

# 如果找不到对应的碱基对，使用默认编码（全 0）
default_7bits = [0, 0, 0, 0, 0, 0, 0]


def get_7bit_encoding(base1, base2):
    """
    返回 base1-base2 的 7bit 编码。
    如果字典中没有定义，则返回 default_7bits。
    """
    pair = (base1.upper(), base2.upper())
    return pair_to_7bits.get(pair, default_7bits)


# =====================================================================
# 2. 定义区域标志位获取函数
# =====================================================================

def get_region_bits(position):
    """
    根据碱基位置获取 2-bit 区域标志位。

    参数：
    - position (int): 碱基位置，1-based。

    返回：
    - list: 2-bit 区域标志位
            01: 简单区域 (1-15 bp)
            11: 种子区域 (16-20 bp)
            10: PAM 区域 (21-23 bp)
    """
    if 1 <= position <= 15:
        return [0, 1]  # 简单区域
    elif 16 <= position <= 20:
        return [1, 1]  # 种子区域
    elif 21 <= position <= 23:
        return [1, 0]  # PAM 区域
    else:
        raise ValueError(f"无效的碱基位置: {position}. 必须在1到23之间。")


# =====================================================================
# 3. 定义将一条脱靶序列与一条 sgRNA 序列转换为 shape=(23, 9) 的矩阵的函数
# =====================================================================

def nine_bit_encode_sequences(off_target_seq, sgrna_seq):
    """
    将脱靶序列和 sgRNA 序列转换为 9bit 编码矩阵。

    每个碱基被编码为 7-bit 向量，然后加上 2-bit 区域标志位。

    参数：
    - off_target_seq (str): 脱靶序列（23bp）
    - sgrna_seq      (str): sgRNA 序列（23bp）

    返回：
    - np.ndarray: shape = (23, 9) 的编码矩阵
    """
    encoded_matrix = []

    for i, (ot_base, sg_base) in enumerate(zip(off_target_seq, sgrna_seq), start=1):
        # 若 sgRNA 碱基为 'N'，则替换为脱靶序列对应碱基
        if sg_base.upper() == 'N':
            sg_base = ot_base.upper()

        # 获取 7-bit 编码
        bits7 = get_7bit_encoding(ot_base, sg_base)

        # 获取区域标志位
        region_bits = get_region_bits(i)

        # 合并为 9-bit 编码
        bits9 = bits7 + region_bits
        encoded_matrix.append(bits9)

    return np.array(encoded_matrix, dtype=np.uint8)  # shape = (23, 9)


# =====================================================================
# 4. 从文件读取数据，并将每条样本编码为 shape=(23, 9)，与标签一起返回
# =====================================================================

def process_file(input_file):
    """
    读取输入文件，对脱靶序列和 sgRNA 序列进行 9bit 编码，并返回编码后的数据和类别标签。

    参数：
    - input_file (str): 输入 TXT 文件路径

    返回：
    - tuple: (encoded_data, labels)
             encoded_data.shape = (N, 23, 9)
             labels.shape       = (N,)
    """
    # 读取 TSV（制表符分隔），且有表头
    df = pd.read_csv(input_file, sep='\t', header=0, dtype=str)

    if df.shape[1] < 3:
        raise ValueError("输入文件至少需要三列：脱靶序列、sgRNA 序列、类别标签。")

    # 提取三列
    off_target_sequences = df.iloc[:, 0].str.upper()
    sgrna_sequences = df.iloc[:, 1].str.upper()
    class_labels = df.iloc[:, 2].astype(int)

    # 检查每条序列长度是否为 23
    valid_length = (off_target_sequences.str.len() == 23) & (sgrna_sequences.str.len() == 23)
    if not valid_length.all():
        invalid_count = (~valid_length).sum()
        print(f"警告：有 {invalid_count} 条记录长度不为 23，将被忽略。")
        off_target_sequences = off_target_sequences[valid_length]
        sgrna_sequences = sgrna_sequences[valid_length]
        class_labels = class_labels[valid_length]

    # 开始编码
    encoded_list = []
    for ot_seq, sg_seq in zip(off_target_sequences, sgrna_sequences):
        matrix_9 = nine_bit_encode_sequences(ot_seq, sg_seq)  # (23, 9)
        encoded_list.append(matrix_9)

    encoded_data = np.stack(encoded_list, axis=0)  # shape = (N, 23, 9)
    labels = class_labels.values  # shape = (N,)

    return encoded_data, labels


# =====================================================================
# 5. 主函数：读取文件 -> 编码 -> 保存 .npz -> 验证
# =====================================================================

def main():
    input_file = 'D:/pycharm/liu_bio/liu_sgrna_final/chaptrer1/project/A-siteseq/new/SITESeq-Dataset.txt'
    output_file = 'D:/pycharm/liu_bio/liu_sgrna_final/chaptrer1/project/A-siteseq/new/9bit_SITESeq-Dataset.npz'

    # 处理文件并获取编码后的数据和类别标签
    try:
        encoded_data, labels = process_file(input_file)
        print(f"成功编码 {encoded_data.shape[0]} 条序列。")
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return

    # 保存到 npz
    try:
        np.savez_compressed(output_file, data=encoded_data, labels=labels)
        print(f"已保存编码结果至 '{output_file}'")
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return

    # 验证保存的数据
    try:
        loaded = np.load(output_file)
        print("\n验证保存的数据：")
        print("包含的数组:", loaded.files)
        print("data 形状:", loaded['data'].shape)
        print("labels 形状:", loaded['labels'].shape)

        print("\ndata 的第一个样本 (前 3 位)：")
        print(loaded['data'][0][:23])  # 显示第一条序列的前 3 个编码位
        print("labels 的第一个标签:", loaded['labels'][0])
    except Exception as e:
        print(f"验证时出错: {e}")


if __name__ == "__main__":
    main()
