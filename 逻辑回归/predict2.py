import pandas as pd
import joblib

# 1. 读取 Test 数据
test_df = pd.read_csv('../data/test.csv')

# 2. 加载之前训练好的模型
model = joblib.load('logistic_regression_model.pkl')

# ==========================================
#  关键步骤：必须对 Test 集做一模一样的特征工程
# ==========================================
def feature_engineering(df):
    df_eng = df.copy()
    # 1. 处理 pdays
    df_eng['was_contacted'] = (df_eng['pdays'] != -1).astype(int)
    # 2. 交互特征
    housing_num = df_eng['housing'].map({'yes': 1, 'no': 0, 'unknown': 0})
    loan_num = df_eng['loan'].map({'yes': 1, 'no': 0, 'unknown': 0})
    df_eng['debt_level'] = housing_num + loan_num
    return df_eng

# 对 test 数据应用特征工程
test_df_eng = feature_engineering(test_df)

# ==========================================
#  核心区别：使用 predict_proba
# ==========================================
# model.predict(X)       -> 输出 [0, 1, 0...] (不要用这个！)
# model.predict_proba(X) -> 输出 [[0.8, 0.2], [0.3, 0.7]...] (二维数组)

# 我们只需要 "y=1" (正类) 的概率，所以取第 2 列 (索引为 1)
test_pred_prob = model.predict_proba(test_df_eng)[:, 1]

# 3. 生成提交 DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'y': test_pred_prob
})

# 4. 预览一下 (确保是小数)
print(submission.head())

# 5. 保存
submission.to_csv('submission.csv', index=False)
print("提交文件 submission.csv 已生成！")