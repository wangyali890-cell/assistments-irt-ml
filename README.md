# ASSISTments + IRT 学生作答预测

基于 ASSISTments 2009–2010 Skill Builder 学习行为数据，结合 Rasch 项目反应理论（IRT）与 XGBoost，预测学生下一次作答正确的概率，并评估心理测量特征是否能够补充普通行为历史特征。

## 1. 项目简介

项目回答的问题是：

> 在学生已有作答历史的前提下，加入 Rasch IRT 估计的学生能力和题目难度后，学生后续作答预测是否有所改善？

项目重点展示：

- Python 数据清洗与特征工程；
- 学生内时间切分和数据泄漏控制；
- Logistic Regression、XGBoost 和概率预测评价；
- Rasch/1PL IRT 的基础实现；
- 心理测量特征与机器学习模型融合；
- 中文项目文档、图表和面试材料。

## 2. 数据集

本项目使用 [ASSISTments 2009–2010 Skill Builder 修正版](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010)。官方说明该版本将多技能题合并为一条学生-题目记录，适合用于学生表现预测。

原始文件应保存为：

```text
data/raw/skill_builder_data_corrected.csv
```

原始文件不提交 Git。当前运行数据规模如下：

| 项目 | 数量 |
|---|---:|
| 原始文件大小 | 约 64.4 MB |
| 交互记录 | 341,879 |
| 学生 | 3,168 |
| 题目 | 26,628 |
| skill | 150 |
| 总体正确率 | 64.53% |

本项目使用的字段：

- `user_id`：学生 ID；
- `problem_id`：题目 ID；
- `correct`：是否首次作答正确；
- `order_id`：交互时间顺序；
- `skill_id`：题目对应的 skill 标签。

## 3. 预测任务

对学生在第 (t) 次交互的作答结果进行预测：

\[
P(Y_{it}=1)
\]

其中 (Y_{it}=1) 表示当前作答正确。评价对象是已有一定学习历史的学生，不是新学生 cold-start 泛化。

## 4. 数据处理与特征

处理流程为：

1. 删除关键字段缺失、标签非法和重复学生-题目记录；
2. 按 `user_id`、`order_id` 稳定排序；
3. 删除总交互数少于 10 的学生；
4. 对每名学生按时间顺序切分为 Train 70%、Validation 15%、Test 15%；
5. 生成只使用当前交互之前信息的历史特征。

行为特征包括：

- 学生历史作答次数；
- 学生历史正确率；
- 最近 5 次正确率；
- 学生在当前 skill 上的历史作答次数；
- 学生在当前 skill 上的历史正确率。

当前运行切分结果为：

| 切分 | 交互数 | 学生数 | 正确率 |
|---|---:|---:|---:|
| Train | 237,936 | 3,168 | 64.98% |
| Validation | 49,760 | 3,168 | 63.75% |
| Test | 54,183 | 3,168 | 63.27% |

## 5. 数据泄漏控制

历史正确率不能使用学生全量正确率，因为全量数据包含当前答案和未来答案。

代码使用累计值减去当前 `correct`，并使用 `shift(1)` 计算最近正确率，因此当前目标不会进入当前特征。

IRT 只使用 Train 交互拟合。得到的学生能力 (	heta) 和题目难度 (b) 在 Validation/Test 阶段固定，不使用未来答案重新估计。Test 中训练集未见过的题目使用训练集平均难度。

## 6. 模型

### Logistic Regression

使用普通行为历史特征作为概率预测基线。

### Rasch IRT

采用 1PL Rasch 模型：

\[
P(Y_{ij}=1)=\sigma(\theta_i-b_j)
\]

- (	heta_i)：学生能力；
- (b_j)：题目难度；
- (	heta_i-b_j)：学生能力与题目难度的匹配程度。

实现上使用稀疏学生/题目指示变量逻辑回归拟合，参数只来自 Train。

### XGBoost

使用固定的轻量参数，不进行大规模超参数搜索。

### XGBoost + IRT

在普通行为特征上加入：

- `theta`；
- `item_difficulty`；
- `theta_minus_difficulty`；
- `irt_probability`。

## 7. Test 集结果

| 模型 | AUC | LogLoss | Brier Score |
|---|---:|---:|---:|
| Logistic Regression | 0.7145 | 0.5853 | 0.1995 |
| Rasch IRT | 0.7454 | 0.5680 | 0.1928 |
| XGBoost | 0.7181 | 0.5780 | 0.1966 |
| XGBoost + IRT | **0.7667** | **0.5666** | **0.1893** |

相较普通 XGBoost：

```text
ΔAUC = +0.0486
ΔLogLoss = -0.0114
ΔBrier = -0.0074
```

## 8. 主要发现

1. XGBoost + IRT 在 Test 集上取得最高 AUC，并同时改善 LogLoss 和 Brier Score。
2. Rasch IRT 单独预测的 AUC 为 0.7454，说明训练阶段估计的学生能力和题目难度包含有用的概率预测信息。
3. IRT 特征带来的提升并不等于因果效果，只能说明在当前切分和特征设计下具有额外预测价值。
4. 学生历史行为与学生能力存在重叠信息，因此不能预先假设 IRT 一定提升所有数据集上的表现。
5. Test 仍然是已有学生的未来作答预测，不能直接推广到完全没有历史的新学生。

## 9. 可解释性与图表

项目生成以下中文图表：

- `artifacts/figures/interaction_count_distribution.png`：学生交互次数分布；
- `artifacts/figures/correctness_distribution.png`：正确与错误比例；
- `artifacts/figures/theta_distribution.png`：学生能力分布；
- `artifacts/figures/item_difficulty_distribution.png`：题目难度分布；
- `artifacts/figures/model_comparison.png`：模型性能比较；
- `artifacts/figures/feature_importance.png`：XGBoost + IRT 特征重要性。

## 10. 项目结构

```text
data/raw/                  原始数据，不提交 Git
data/processed/            清洗后的交互和特征
notebooks/01_eda.ipynb     中文探索性分析 Notebook
src/preprocess.py          数据清洗和时间切分
src/features.py            causal 历史特征
src/irt.py                 Rasch IRT
src/train.py               四个模型的训练和输出
src/evaluate.py            指标和图表
artifacts/results.csv      模型结果
artifacts/figures/         中文图表
tests/test_core.py         核心 sanity checks
resume_bullets.md          中文简历要点
interview_notes.md         中文面试准备
```

## 11. 如何运行

安装依赖：

```bash
python -m pip install -r requirements.txt
```

将官方修正版数据保存到 `data/raw/skill_builder_data_corrected.csv` 后，运行完整流程：

```bash
python -m src.train
```

单独运行预处理：

```bash
python -m src.preprocess --input data/raw/skill_builder_data_corrected.csv --output data/processed/interactions.csv
```

运行测试：

```bash
python -m pytest -q
```

打开 Notebook：

```bash
jupyter notebook notebooks/01_eda.ipynb
```

## 12. 项目局限

- 评价对象是已有作答历史的学生，不是 cold-start 新学生；
- IRT 使用 Rasch/1PL，未实现 2PL、3PL 或动态 IRT；
- Test 阶段固定训练阶段估计的 (	heta) 和 (b)；
- 没有建模随时间变化的知识状态；
- ASSISTments 数据存在特定教学场景、作答顺序和 skill 标注限制；
- 结果是预测关联，不代表 IRT 特征对正确率有因果影响。

## 13. 后续方向

- 2PL IRT；
- 动态 IRT；
- Bayesian Knowledge Tracing；
- LSTM 或 Transformer Knowledge Tracing；
- 更严格的新学生泛化实验。
