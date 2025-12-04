from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# 读取训练集数据
# 注意：这里的路径 'playground-series-s5e8.zip/train.csv' 需要根据你实际文件的位置修改
train_df = pd.read_csv('../data/train.csv')

# --- 1. 准备数据 ---
X = train_df.drop(columns=['id', 'y']) 
y = train_df['y']

# --- 2. 定义预处理逻辑 ---
# 自动识别列类型
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 数值列处理：中位数填充缺失值 + 标准化
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())
])

# 类别列处理：最频繁值填充缺失值 + One-Hot编码 (遇到新类别忽略)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')), # 保持 unknown 为一类
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 组合起来
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 3. 定义逻辑回归模型 ---
# max_iter=2000 保证有足够时间收敛，C=1.0 是正则化强度默认值
model = LogisticRegression(max_iter=2000, random_state=42, C=1.0, solver='lbfgs')

# --- 4. 构建最终管道 ---
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])

# --- 5. 执行 12折交叉验证 ---
print("开始执行 12-Fold Cross Validation (这可能需要几秒钟)...")

# 实例化切分器
cv = StratifiedKFold(n_splits=12, shuffle=True, random_state=42)

# 计算 AUC
scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

# --- 6. 输出结果 ---
print(f"\n每折的 AUC 分数:\n{scores}")
print(f"\n>>> 平均 AUC: {scores.mean():.5f} (标准差: {scores.std():.5f})")