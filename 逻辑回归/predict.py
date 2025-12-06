import pandas as pd
import joblib

# --- 1. 加载保存的模型 ---
print("加载模型...")
model_path = 'logistic_regression_model.pkl'
clf = joblib.load(model_path)
print("模型加载完成！")

# --- 2. 读取测试集数据 ---
print("\n读取测试集数据...")
test_df = pd.read_csv('../data/test.csv')
print(f"测试集样本数: {len(test_df)}")

# --- 3. 准备特征数据 ---
# 保存id列用于最后的提交
test_ids = test_df['id']

# 删除id列，保留特征列
X_test = test_df.drop(columns=['id'])

# --- 4. 进行预测 ---
print("\n开始预测...")
# 使用 predict 获取预测类别 (0 或 1)
y_pred = clf.predict(X_test)
print("预测完成！")

# --- 5. 创建提交文件 ---
print("\n创建提交文件...")
submission = pd.DataFrame({
    'id': test_ids,
    'y': y_pred
})

# 保存为CSV文件
submission_path = 'submission.csv'
submission.to_csv(submission_path, index=False)
print(f"\n提交文件已保存到: {submission_path}")
print(f"预测结果示例:")
print(submission.head(10))
