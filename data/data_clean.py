import pandas as pd
import numpy as np

# 读取数据
train_df = pd.read_csv('./train.csv')

# 1. 看看数据的基本信息
print("数据总行数:", len(train_df))
print("\n数据类型概览:")
print(train_df.info())

# 2. 检查是否有缺失值 (NaN)
print("\n缺失值统计:")
print(train_df.isnull().sum())

# 3. 检查是否有 'unknown' 这种隐形缺失值（银行数据里常有）
# 我们挑几个类别列看看
categorical_cols = train_df.select_dtypes(include=['object']).columns
print("\n类别特征中的 'unknown' 值统计:")
for col in categorical_cols:
    unknown_count = (train_df[col] == 'unknown').sum()
    if unknown_count > 0:
        print(f"{col}: {unknown_count} 个 unknown")