# 第一轮 12-11-18:50

通过分析前三名所使用的方法，不约而同地提到了一个关键词，AutoGluon

我们先来了解一下这是什么

## 1）AutoGluon 是什么

**AutoGluon** 是一个面向实际业务/竞赛的 **AutoML（自动机器学习）框架**，核心目标是：

> 你只需要给它一份表格数据（特征 + 标签）和评价指标，它就能自动完成“训练很多模型 → 选最好 → 还做集成提升效果”。

它最典型的能力包括：

- **自动训练多种模型**：不用你手动挑“用 XGBoost 还是 CatBoost 还是神经网络”，它会并行/串行尝试很多模型体系。
- **自动模型选择与调参**：在时间预算内做合理的搜索，而不是你一个个参数试。
- **自动集成（Ensemble）**：把多个模型组合起来（Bagging/Stacking/加权融合等），通常比单模型更强更稳。
- **训练全流程封装**：从训练、验证、保存模型、输出排行榜，到最终预测，都可以一套 API 搞定。



### Step A：准备训练数据（AutoGluon 的输入格式）

代码先读入 `train.csv / test.csv`，然后做一些处理（这里我们不细讲特征工程），最终得到：

- `train_data`：训练数据（包含特征 + 标签列 `y`）
- `test_data`：测试数据（只有特征，没有 `y`）
- `label='y'`：告诉 AutoGluon 哪一列是要预测的目标

**关键点**：AutoGluon 的 `TabularPredictor.fit()` 期待的就是这种“表格 DataFrame”，并且通过 `label` 指定目标列。

------

### Step B：创建 `TabularPredictor`（定义任务）

```

predictor = TabularPredictor(
    label=label,
    eval_metric='roc_auc',
    problem_type='binary',
    path=model_path
)
```

这一段相当于在告诉 AutoGluon：

- 这是一个 **二分类任务**（`problem_type='binary'`）
- 评价指标用 **ROC-AUC**（`eval_metric='roc_auc'`）
- 模型与训练产物保存到 **path 目录**（`ag_models_ultimate_run`）

你可以把 `TabularPredictor` 理解为：
 **“AutoGluon 的任务管理器/训练器对象”**，后面所有训练与预测都通过它完成。

------

### Step C：`fit()` 启动自动训练（核心）

```

.fit(
    train_data,
    presets='best_quality',
    time_limit=TIME_LIMIT,
    fit_weighted_ensemble=True,
    ag_args_ensemble=ag_args_ensemble,
    num_gpus=1
)
```

这是整段代码的“AutoML 核心开关”，它做的事可以概括为：

1. **按 `best_quality` 策略训练很多模型**
    `presets='best_quality'` 表示“尽量追求效果，不惜训练成本”。
    AutoGluon 会训练多类模型，并且更积极地做集成（这就是为什么它适合冲分，但也更耗时）。
2. **遵守时间预算**
    `time_limit=24h`：AutoGluon 在这个总时间内尽量训练更多更强的组合。
3. **强制更强的集成策略**
    `ag_args_ensemble` 里指定了：

- `num_folds=10`：做更稳的交叉验证式 Bagging
- `num_stack_levels=3`：做更深的 Stacking（堆叠）

1. **最终做加权融合**
    `fit_weighted_ensemble=True`：训练一个“加权投票/加权融合”的总模型，把多个模型结果再融合一次，进一步榨取分数。
2. **允许用 GPU**
    `num_gpus=1`：允许某些模型/模块使用 GPU 加速（取决于环境和具体模型）。

> 总结一句：`fit()` 这一步就是 **“给 AutoGluon 一份训练表，让它在 24h 内自动训练 + 自动集成 + 自动挑最强方案”**。

```
import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
import os

# ==========================================
# 1. 特征工程 (保留精华)
# ==========================================
def feature_engineering(df):
    df_eng = df.copy()
    
    # --- 核心修复 ---
    df_eng['was_contacted'] = (df_eng['pdays'] != -1).astype(int)
    df_eng['pdays'] = df_eng['pdays'].replace(-1, 0)
    
    # --- 强力业务特征 ---
    # 字符串组合特征 (AutoGluon 的 CatBoost/NN 非常喜欢)
    df_eng['housing_loan_combo'] = df_eng['housing'].astype(str) + "_" + df_eng['loan'].astype(str)
    df_eng['job_edu_combo'] = df_eng['job'].astype(str) + "_" + df_eng['education'].astype(str)
    df_eng['poutcome_contact_combo'] = df_eng['poutcome'].astype(str) + "_" + df_eng['contact'].astype(str)
    
    # --- 数值处理 ---
    df_eng['duration_log'] = np.log1p(df_eng['duration'])
    df_eng['balance_log'] = np.sign(df_eng['balance']) * np.log1p(np.abs(df_eng['balance']))
    
    # 交互比率
    df_eng['duration_campaign_ratio'] = df_eng['duration'] / (df_eng['campaign'] + 1)
    df_eng['pdays_previous_ratio'] = df_eng['pdays'] / (df_eng['previous'] + 1)
    
    return df_eng

# ==========================================
# 2. 数据准备
# ==========================================
print("📦 读取数据...")
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

print("🛠️  应用特征工程...")
train_df_eng = feature_engineering(train_df)
test_df_eng = feature_engineering(test_df)

train_data = train_df_eng.drop(columns=['id'])
test_data = test_df_eng.drop(columns=['id'])
label = 'y'

# ==========================================
# 3. 极限配置 (自动托管模式)
# ==========================================
# 设定运行时间：24 小时 (86400 秒)
# AutoGluon 会在这个时间内尽可能多地训练模型
TIME_LIMIT = 3600 * 24 

model_path = 'ag_models_ultimate_run'

print(f"\n🚀 启动 AutoGluon 极限模式...")
print(f"⏱️  计划运行时间: {TIME_LIMIT / 3600} 小时")
print("🔥 策略: 全自动 best_quality + 3层堆叠")

# 我们只通过 ag_args_ensemble 告诉它：我们要更深的网络，更稳的验证
# 其他的模型选择完全交给 best_quality
ag_args_ensemble = {
    'num_folds': 10,       # 强制 10 折交叉验证 (更稳)
    'num_stack_levels': 3, # 强制 3 层堆叠 (上限更高)
}

predictor = TabularPredictor(
    label=label,
    eval_metric='roc_auc',
    problem_type='binary',
    path=model_path
).fit(
    train_data,
    # 'best_quality' 是核心：它会自动包含 GBM, CAT, XGB, XT, RF, NN_TORCH, FASTAI
    # 并且会自动进行 Bagging 和 Stacking
    presets='best_quality', 
    time_limit=TIME_LIMIT,
    # 传入堆叠参数
    fit_weighted_ensemble=True,
    ag_args_ensemble=ag_args_ensemble,
    num_gpus=1
)

# ==========================================
# 4. 结果分析与预测
# ==========================================
print("\n🏆 训练结束！生成排行榜...")
# 打印排行榜
leaderboard = predictor.leaderboard(train_data, silent=True)
leaderboard.to_csv("leaderboard_ultimate.csv")
print(leaderboard.head(20)) # 看看前20个模型是谁

print("\n📝 生成最终预测...")
# AutoGluon 自动选择验证集分数最高的模型来预测
y_pred_proba = predictor.predict_proba(test_data)
final_preds = y_pred_proba[1]

submission = pd.DataFrame({
    'id': test_df['id'],
    'y': final_preds
})

submission.to_csv('submission_ultimate_24h.csv', index=False)
print("🎉 submission_ultimate_24h.csv 已生成！")
```

![image-20251212193641620](https://image-hub.oss-cn-chengdu.aliyuncs.com/image-20251212193641620.png)

![image-20251212195338875](https://image-hub.oss-cn-chengdu.aliyuncs.com/image-20251212195338875.png)

# 第二轮

先对原始 train/test 做比较重的特征工程 + Target Encoding → 用 AutoGluon 训练一个 Stage 1 模型 → 利用该模型的 OOF 预测/测试预测作为新的“高阶特征（stage1_pred）”注入数据 → 把增强后的数据导出为 CSV，供第二个脚本（Stage 2）继续建模。

**特征工程模块**：
 将原始训练集与测试集合并，在统一框架下进行缺失/异常值修复、组合类别特征构造、数值变换与按类别的统计特征提取，从而丰富输入变量的表达能力。

**Target Encoding 模块**：
 对职业、教育等关键类别变量采用基于 10 折交叉验证的 Target Encoding，将类别映射为稳健的历史转化率特征，有效缓解高基数类别带来的稀疏问题，同时通过折外编码避免信息泄露。

**Stage 1 AutoML 预训练模块**：
 利用 AutoGluon Tabular 以 `best_quality` 预设在两小时预算内训练一个高性能二分类模型，自动集成多种基础学习器，并保存在指定路径以便复用。

**递归特征注入模块**：
 通过 Stage 1 模型对训练集产生 OOF 概率预测，对测试集产生常规概率预测，并将正类概率作为新的高阶特征 `stage1_pred` 注入数据，使后续 Stage 2 模型能够显式利用“前一阶段模型的风险评分”。

**中间结果持久化模块**：
 将增强后的训练集与测试集导出为 `train_enhanced.csv` 和 `test_enhanced.csv`，为第二阶段更复杂的建模流程提供稳定输入，避免后续脚本崩溃时重复耗时的预处理与 Stage 1 训练。

```
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

# ================= 配置 =================
# 增强数据路径 (必须和脚本1一致)
train_file = 'train_enhanced.csv'
test_file = 'test_enhanced.csv'
path_stage2 = 'ag_models_stage2_ultimate'

# 运行时间: 24小时 (86400秒) - 强度拉满
TIME_LIMIT = 3600 * 24 

# ================= 主流程 =================
if __name__ == '__main__':
    print(f"📦 读取增强后的数据: {train_file}...")
    try:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
    except FileNotFoundError:
        print("❌ 错误：未找到数据文件！请先运行脚本 1 生成数据。")
        exit()

    label = 'y'
    print(f"✅ 数据加载成功。特征维度: {train_data.shape[1]} (包含 stage1_pred)")
    
    print("\n🚀 [Stage 2] 启动 AutoGluon 终极模式 (Nuclear Option)...")
    print(f"⏱️  计划运行时间: {TIME_LIMIT / 3600} 小时")
    print("🔥 策略: 3层堆叠 (L3) + 10折 Bagging")

    # 极限参数配置
    ag_args_ensemble = {
        'num_folds': 10,       # 10折交叉验证
        'num_stack_levels': 3, # 3层堆叠
    }

    predictor_final = TabularPredictor(
        label=label,
        eval_metric='roc_auc',
        problem_type='binary',
        path=path_stage2
    ).fit(
        train_data,
        presets='best_quality', # 自动开启 Bagging/Stacking
        time_limit=TIME_LIMIT,
        ag_args_ensemble=ag_args_ensemble,
        num_gpus=1
    )

    # ================= 收尾 =================
    print("\n🏆 训练结束！")
    leaderboard = predictor_final.leaderboard(train_data, silent=True)
    leaderboard.to_csv("leaderboard_sota_ultimate.csv")
    print(leaderboard.head(20))

    print("\n📝 生成最终提交文件...")
    # 自动选择验证集分数最高的模型进行预测
    y_pred_final = predictor_final.predict_proba(test_data)[predictor_final.positive_class]

    # 读取原始ID用于生成提交文件
    test_raw = pd.read_csv('./data/test.csv')
    submission = pd.DataFrame({
        'id': test_raw['id'],
        'y': y_pred_final
    })

    submission.to_csv('submission_sota_ultimate.csv', index=False)
    print("🎉 submission_sota_ultimate.csv 已生成！祝霸榜！")
```

### 成绩更差了，为什么？

### 1. “原生”胜过“加工”：AutoGluon 的处理机制

- **你的做法**：手动做了 Target Encoding，把 `job`（职业）变成了 `0.08`（违约率）这样的数字。
- **AutoGluon 的做法**：它内部集成的 **CatBoost** 和 **LightGBM** 对类别特征有极其强大的原生处理能力。
  - **CatBoost** 的名字就是 Categorical Boosting，它内部有一套极其复杂的动态编码机制（Ordered Target Statistics），**比我们手动写的 K-Fold Target Encoding 要强得多**。
  - **后果**：当你手动把 `job` 变成数字后，CatBoost 就“瞎了”，它只能看到一个普通的浮点数，失去了使用它内部黑科技的机会。**你相当于把一块顶级的和牛（原始类别）做成了牛肉丸（数字），然后再交给米其林大厨（CatBoost），大厨反而没法发挥了。**

### 2. 递归特征的“过拟合陷阱” (Leakage)

- 我们在第一版中生成了 `stage1_pred`（递归特征）。
- 虽然我们用了 OOF（Out-of-Fold）来防止泄露，但在 AutoGluon 的 `best_quality` 模式下，它内部又会进行 10 折交叉验证。
- **风险**：这种**“堆叠套堆叠”**极容易导致**数据泄露**。模型会发现 `stage1_pred` 这个特征太准了，于是它放弃了学习其他特征（如 balance, age），变得极度依赖这个特征。
- **结果**：在训练集和验证集上分数可能很高，但一旦到了测试集（Test），如果这个特征稍微有点偏差，整个模型就会崩盘。

### 3. 信息压缩带来的损失

- **Target Encoding** 是有损压缩。
- 假设 `Student`（学生）和 `Retired`（退休）的违约率都是 `0.1`。
- 如果你用了 Target Encoding，这两个职业都会变成 `0.1`。在模型眼里，学生和退休老人就**变成同一种人了**。
- 但实际上，学生可能没钱但未来潜力大，老人是有钱但不想折腾。保留原始字符串，模型可以通过与其他特征（如 `age`）交叉来区分他们；一旦变成了 `0.1`，这种区分能力就丢失了。

### 4. 10折 x 3层堆叠 > 人工特征

- 你最后贴的这一版代码，虽然特征工程简单，但你开启了 **`num_folds=10`** 和 **`num_stack_levels=3`**。
- 这相当于你组建了一个由几十个模型组成的“超级评审团”。
- AutoGluon 的强项就是**模型融合（Ensembling）**。只要给它足够纯净、未被过度破坏的原始数据，它内部的神经网络（NN_TORCH）和树模型（XGB/CAT）能自动挖掘出比人类手动构造更复杂的非线性关系。
