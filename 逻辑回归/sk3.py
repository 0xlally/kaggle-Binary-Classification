import pandas as pd
import numpy as np
import joblib
import contextlib
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer, PolynomialFeatures, FunctionTransformer, PowerTransformer
from sklearn.impute import SimpleImputer
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

# 忽略警告，保持输出清爽
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
#  核心组件：自定义 GPU 逻辑回归
# ==========================================
class PyTorchLogReg(ClassifierMixin, BaseEstimator):
    # 【关键修复】强制声明自己是分类器
    _estimator_type = "classifier"

    def __init__(self, C=1.0, penalty='l2', class_weight=None, max_iter=2000, lr=0.05, device=None):
        self.C = C
        self.penalty = penalty
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.lr = lr
        self.device = device  # 延迟到 fit 时处理
        self.model = None
        self.classes_ = None  # 【关键修复】必须初始化
        self.n_features_in_ = None

    def fit(self, X, y):
        # 0. 确定设备
        if self.device is None:
            fit_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            fit_device = self.device

        # 1. Sklearn 标准检查
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        
        # 【关键修复】设置 classes_ 属性，防止报错
        self.classes_ = unique_labels(y)

        # 2. 处理 Class Weight
        pos_weight = None
        if self.class_weight == 'balanced':
            num_neg = (y == 0).sum()
            num_pos = (y == 1).sum()
            if num_pos > 0:
                weight_val = num_neg / num_pos
                pos_weight = torch.tensor(weight_val, dtype=torch.float32).to(fit_device)

        # 3. 数据转 Tensor (全量入显存)
        X_t = torch.tensor(X, dtype=torch.float32).to(fit_device)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(fit_device)

        # 4. 定义模型
        self.model = nn.Linear(self.n_features_in_, 1).to(fit_device)
        
        # 5. Loss & Optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 6. 训练循环
        self.model.train()
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            outputs = self.model(X_t)
            loss = criterion(outputs, y_t)
            
            # 手动实现正则化
            reg_loss = 0.0
            if self.penalty == 'l2':
                for param in self.model.parameters():
                    reg_loss += torch.sum(param ** 2)
                loss = loss + (1 / (2 * self.C)) * reg_loss
            elif self.penalty == 'l1':
                for param in self.model.parameters():
                    reg_loss += torch.sum(torch.abs(param))
                loss = loss + (1 / self.C) * reg_loss
            
            loss.backward()
            optimizer.step()
            
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        
        if self.device is None:
            pred_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            pred_device = self.device
            
        X = check_array(X, accept_sparse=False)
        X_t = torch.tensor(X, dtype=torch.float32).to(pred_device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            prob_pos = torch.sigmoid(logits).cpu().numpy()
            
        return np.hstack([1 - prob_pos, prob_pos])

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

# ==========================================
#  辅助工具：进度条
# ==========================================
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# ==========================================
#  1. 读取数据
# ==========================================
print("📦 读取数据...")
# 请根据实际情况修改路径
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

# ==========================================
#  2. 特征工程 (Pandas)
# ==========================================
def feature_engineering(df):
    df_eng = df.copy()
    df_eng['was_contacted'] = (df_eng['pdays'] != -1).astype(int)
    housing_num = df_eng['housing'].map({'yes': 1, 'no': 0, 'unknown': 0})
    loan_num = df_eng['loan'].map({'yes': 1, 'no': 0, 'unknown': 0})
    df_eng['debt_level'] = housing_num + loan_num
    df_eng['duration_log'] = np.log1p(df_eng['duration'])
    return df_eng

print("🛠️  正在进行基础特征工程...")
train_df_eng = feature_engineering(train_df)
test_df_eng = feature_engineering(test_df)

X_raw = train_df_eng.drop(columns=['id', 'y']) 
y = train_df_eng['y']

# ==========================================
#  3. 预处理 (Sklearn Pipeline) - 移至循环外
# ==========================================
print("⚙️  正在进行 CPU 密集型预处理 (Yeo-Johnson & Polynomial)...")
print("   (这步只需做一次，大概 1-2 分钟，请耐心等待...)")

skewed_features = ['balance', 'duration', 'campaign', 'previous']
skewed_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('power', PowerTransformer(method='yeo-johnson'))
])

poly_features = ['day', 'pdays', 'age'] 
poly_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)) 
])

age_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('binning', KBinsDiscretizer(
        n_bins=10, 
        encode='onehot', 
        strategy='quantile', 
        quantile_method='averaged_inverted_cdf' # 【修复警告】
    ))
])

categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('skewed', skewed_transformer, skewed_features),
        ('poly', poly_transformer, poly_features),
        ('age_bin', age_transformer, ['age']),
        ('cat', categorical_transformer, categorical_features),
        ('rest', StandardScaler(), ['debt_level', 'was_contacted'])
    ])

# 先 fit_transform 训练集
X_processed = preprocessor.fit_transform(X_raw)
# 再 transform 测试集
X_test_processed = preprocessor.transform(test_df_eng)

print(f"✅ 预处理完成！特征矩阵形状: {X_processed.shape}")

# ==========================================
#  4. Grid Search (GPU 版)
# ==========================================

# 这里的 Pipeline 只包含分类器，因为数据已经处理好了
pipeline = Pipeline(steps=[
    ('classifier', PyTorchLogReg(max_iter=3000, lr=0.05))
])

param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10], 
    'classifier__class_weight': ['balanced', None],
    'classifier__penalty': ['l1', 'l2'] 
}

n_candidates = len(list(ParameterGrid(param_grid)))
n_folds = 12
total_fits = n_candidates * n_folds

print(f"\n🚀 开始 Grid Search (纯 GPU 训练，总共 {total_fits} 次)...")

grid_search = GridSearchCV(pipeline, param_grid, cv=n_folds, scoring='roc_auc', n_jobs=-1, verbose=1)

with tqdm_joblib(tqdm(desc="Grid Search Progress", total=total_fits)) as progress_bar:
    # 传入处理好的数据
    grid_search.fit(X_processed, y)

print(f"\n🏆 最佳参数: {grid_search.best_params_}")
print(f"🏆 最佳验证集 AUC: {grid_search.best_score_:.5f}")

# ==========================================
#  5. 生成预测
# ==========================================
best_model = grid_search.best_estimator_

print("\n📝 生成预测文件...")
test_pred_prob = best_model.predict_proba(X_test_processed)[:, 1]

submission = pd.DataFrame({'id': test_df['id'], 'y': test_pred_prob})
submission.to_csv('submission_final_gpu.csv', index=False)
print("🎉 submission_final_gpu.csv 已生成！")