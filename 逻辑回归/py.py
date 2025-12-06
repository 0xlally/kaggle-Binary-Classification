import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# --- 1. 数据准备与预处理 (这一步还得靠 Sklearn) ---

# 读取数据
try:
    train_df = pd.read_csv('../data/train.csv')
except FileNotFoundError:
    print("请检查文件路径")
    # train_df = pd.read_csv('你的路径/train.csv')

# 分离特征和标签
X = train_df.drop(columns=['id', 'y'])
y = train_df['y'].values # 转成 numpy 数组

# 划分训练集和验证集 (为了演示 PyTorch 流程，我们先用简单的 80/20 切分，不搞复杂的交叉验证)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 定义预处理管道 (必须把文字转数字，并把数字归一化)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()) # PyTorch 对数值范围非常敏感，必须标准化！
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ]
)

# 执行预处理
print("正在清洗并转换数据...")
X_train = preprocessor.fit_transform(X_train_raw)
X_val = preprocessor.transform(X_val_raw)

# --- 关键步骤：转换为 PyTorch 张量 (Tensor) ---
# PyTorch 默认使用 float32，而 pandas 默认是 float64
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) # 变成 (N, 1) 形状

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

print(f"输入特征数量: {X_train.shape[1]}")

# --- 2. 搭建 PyTorch 逻辑回归模型 ---

class LogisticRegressionPyTorch(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionPyTorch, self).__init__()
        # 逻辑回归本质：一个线性层 (y = wx + b)
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        # 经过线性层后，用 Sigmoid 压缩到 0-1 之间作为概率
        outputs = torch.sigmoid(self.linear(x))
        return outputs

# 初始化模型
input_dim = X_train.shape[1]
model = LogisticRegressionPyTorch(input_dim)

# 定义损失函数 (二分类通常用 BCELoss: Binary Cross Entropy Loss)
criterion = nn.BCELoss()

# 定义优化器 (Adam 通常比 SGD 收敛更快)
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 3. 手动编写训练循环 ---

num_epochs = 100 # 训练 100 轮
print(f"\n开始使用 PyTorch 训练 {num_epochs} 轮...")

for epoch in range(num_epochs):
    # A. 前向传播 (Forward Pass)
    # 算出预测值
    outputs = model(X_train_tensor)
    # 计算损失 (预测值 vs 真实值)
    loss = criterion(outputs, y_train_tensor)
    
    # B. 反向传播 (Backward Pass)
    optimizer.zero_grad() # 1. 清空以前的梯度 (非常重要！)
    loss.backward()       # 2. 计算新的梯度
    optimizer.step()      # 3. 更新权重
    
    # C. 每 10 轮打印一次进度
    if (epoch+1) % 10 == 0:
        # 在验证集上简单评估一下
        with torch.no_grad(): # 评估时不需要计算梯度，节省内存
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            # 计算 AUC
            val_auc = roc_auc_score(y_val, val_outputs.numpy())
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val AUC: {val_auc:.4f}')

# --- 4. 最终评估 ---
print("\n训练结束。")
model.eval() # 切换到评估模式
with torch.no_grad():
    y_pred_prob = model(X_val_tensor).numpy()
    final_auc = roc_auc_score(y_val, y_pred_prob)
    print(f"PyTorch 逻辑回归最终验证集 AUC: {final_auc:.5f}")