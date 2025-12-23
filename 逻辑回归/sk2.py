import pandas as pd
import numpy as np
import joblib  # 核心库：用于保存和加载模型
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer

from custom_logistic import CustomLogisticRegression

# 1. 读取数据
# 请确保路径正确
try:
    train_df = pd.read_csv('../data/train.csv')
except FileNotFoundError:
    # 备用路径，防止报错
    train_df = pd.read_csv('playground-series-s5e8.zip/train.csv')

# ==========================================
#  函数：特征工程 (Feature Engineering)
#  注意：这个函数以后对 test.csv 也要用！
# ==========================================
def feature_engineering(df):
    df_eng = df.copy()
    
    # 1. 处理 pdays (-1 代表从未联系)
    # 先生成标记特征
    df_eng['was_contacted'] = (df_eng['pdays'] != -1).astype(int)
    
    # 【关键修改】处理“毒数据”
    # 将 -1 替换为 0。这样 pdays 就变成了纯粹的“天数”含义。
    # 逻辑回归会学到：如果 was_contacted=0，pdays的权重不重要；
    # 如果 was_contacted=1，pdays 越大越...
    df_eng['pdays'] = df_eng['pdays'].replace(-1, 0)
    
    # 2. 交互特征: 总负债状况
    housing_num = df_eng['housing'].map({'yes': 1, 'no': 0, 'unknown': 0})
    loan_num = df_eng['loan'].map({'yes': 1, 'no': 0, 'unknown': 0})
    df_eng['debt_level'] = housing_num + loan_num
    
    return df_eng

# 对训练集应用特征工程
print("正在进行特征工程 (已修复 pdays -1 问题)...")
train_df_eng = feature_engineering(train_df)

# ==========================================
#  准备数据 X 和 y
# ==========================================
X = train_df_eng.drop(columns=['id', 'y']) 
y = train_df_eng['y']

# 定义列名 (根据特征工程后的新列调整)
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'debt_level', 'was_contacted']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# ==========================================
#  构建 Pipeline
# ==========================================
# 数值处理
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 年龄分箱 (把年龄变成类别，捕捉非线性)
# 加上 quantile_method 消除警告
age_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('binning', KBinsDiscretizer(
        n_bins=10, 
        encode='onehot', 
        strategy='quantile', 
        quantile_method='averaged_inverted_cdf'
    ))
])

# 类别处理
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 组合处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('age_bin', age_transformer, ['age']), # 单独处理 age
        ('num', numeric_transformer, [c for c in numeric_features if c != 'age']), # 其他数值
        ('cat', categorical_transformer, categorical_features) # 类别
    ])

# 定义自实现逻辑回归模型 (带 class_weight='balanced')
model = CustomLogisticRegression(
    lr=0.05,
    max_iter=3000,
    C=0.1,
    penalty='l2',
    class_weight='balanced',
    fit_intercept=True,
    tol=1e-4,
)

# 最终管道
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])

# ==========================================
#  步骤 1: 交叉验证 (评估模型好坏)
# ==========================================
print("正在执行 12-Fold Cross Validation 评估...")
cv = StratifiedKFold(n_splits=12, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"每折 AUC: {scores}")
print(f">>> 平均 AUC: {scores.mean():.5f} (标准差: {scores.std():.5f})")

# ==========================================
#  步骤 2: 全量训练 (生成最终模型)
# ==========================================
print("\n正在使用所有训练数据训练最终模型...")
clf.fit(X, y)
print("模型训练完成！")

# ==========================================
#  步骤 3: 保存模型
# ==========================================
model_filename = 'custom_logistic_regression_optimized.pkl'
joblib.dump(clf, model_filename)
print(f"模型已成功保存为: {model_filename}")

# ==========================================
#  附加步骤: 演示如何读取模型并生成提交文件
# ==========================================
print("\n--- 模拟预测 Test 集流程 ---")

# 1. 读取 Test 数据
try:
    test_df = pd.read_csv('../data/test.csv')
except FileNotFoundError:
    test_df = pd.read_csv('playground-series-s5e8.zip/test.csv')

# 2. 重要！必须对 Test 集做同样的特征工程
test_df_eng = feature_engineering(test_df)

# 3. 加载模型 (其实可以直接用上面的 clf，这里为了演示加载流程)
loaded_model = joblib.load(model_filename)

# 4. 预测概率 (注意用 predict_proba 获取概率，取第2列即为正类概率)
# 不需要再手动调预处理，loaded_model 里的 pipeline 会自动处理
test_pred_prob = loaded_model.predict_proba(test_df_eng)[:, 1]

# 5. 生成提交 DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'y': test_pred_prob
})

print("预测完成，前 5 行预览：")
print(submission.head())

submission.to_csv('submission4.csv', index=False)
print("submission4.csv 已生成！")