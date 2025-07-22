# 预测出来的脱靶 DNA 变成全大写（第四列）
import pandas as pd
import os
from datetime import datetime


def process_data(input_file):
    # 读取文件
    df = pd.read_csv(input_file, sep='\t', header=None)

    # 将第四列转换为大写
    df[3] = df[3].str.upper()

    # 创建输出文件名
    input_dir = os.path.dirname(input_file)
    input_filename = os.path.basename(input_file)
    output_filename = f"processed_{input_filename}"
    output_path = os.path.join(input_dir, output_filename)

    # 保存处理后的文件
    df.to_csv(output_path, sep='\t', header=False, index=False)
    print(f"处理完成！新文件已保存为: {output_path}")


if __name__ == "__main__":
    # 获取当前日期
    current_date = datetime.now().strftime("%Y-%m-%d")

    # 构建输入文件路径
    input_file = f"/Users/liuhengshen/PycharmProjects/sgrna-final/prediction/step1/data/out.txt"

    # 检查文件是否存在
    if os.path.exists(input_file):
        process_data(input_file)
    else:
        print(f"错误：找不到文件 {input_file}")