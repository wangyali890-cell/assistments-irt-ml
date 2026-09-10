# ASSISTments + IRT 一周 AI 求职项目：Codex 总提示词（中文输出版）

你现在是这个项目的机器学习工程师和项目执行负责人。请帮助我从零开始完成一个**小型、完整、适合放在 GitHub 和简历上的 AI 求职项目**。

这是一个大约 **3–7 天可以完成的 MVP 项目**。

项目最重要的原则是：

**小而完整，不追求复杂。**

不要主动增加 Transformer、LSTM、FastAPI、Docker、数据库、复杂前端、MLOps、云部署等内容，除非核心项目已经全部完成且还有明显余力。

---

# 一、统一语言要求

本项目除代码本身外，**所有面向用户、招聘者、GitHub 阅读者的文字材料默认全部使用中文**。

具体包括但不限于：

- `README.md`：必须使用中文撰写；
- `resume_bullets.md`：必须使用中文撰写；
- `interview_notes.md`：必须使用中文撰写；
- 项目结果总结：使用中文；
- 模型说明：使用中文；
- 数据说明：使用中文；
- 图表标题、坐标轴标签和图注：优先使用中文；
- 项目阶段总结：使用中文；
- 代码注释：优先使用中文；
- docstring：优先使用中文；
- Notebook 中的 Markdown 说明：必须使用中文；
- GitHub 项目介绍文字：使用中文；
- 结果解释与局限性：使用中文。

以下内容可以保留英文：

- Python 变量名；
- 函数名；
- 类名；
- 文件名；
- 库名；
- 模型标准名称；
- 技术缩写，例如 AUC、IRT、XGBoost、SHAP；
- 必须保持英文的 API 或软件参数。

不要为了“国际化”自动把 README、简历要点或面试材料写成英文。

本项目的默认展示语言是：

\[
\boxed{\text{中文}}
\]

---

# 二、项目目标

使用公开的 **ASSISTments** 教育数据，构建一个学生后续作答正确率预测项目，并加入我的专业特色：

**项目反应理论（Item Response Theory, IRT）**

项目核心问题是：

> 加入心理测量模型产生的学生能力和题目难度信息后，机器学习模型的学生作答预测是否有所改善？

这不是学术论文，也不要求方法创新。

目标是向招聘者展示我具备：

- Python 数据处理；
- 特征工程；
- 机器学习；
- 模型评价；
- 基本心理测量建模；
- IRT 与机器学习融合；
- 防止数据泄漏；
- GitHub 项目整理；
- 技术结果解释。

---

# 三、最终项目不要超过这个范围

只需要完成以下四类模型：

## 模型 1：Logistic Regression

使用普通历史行为特征。

## 模型 2：XGBoost

使用普通历史行为特征。

## 模型 3：Rasch / 1PL IRT

单独使用 IRT 预测：

\[
P(Y_{ij}=1)
=
\sigma(\theta_i-b_j)
\]

其中：

- \(\theta_i\)：学生能力；
- \(b_j\)：题目难度。

## 模型 4：XGBoost + IRT

在普通行为特征基础上加入：

\[
\theta_i
\]

\[
b_j
\]

\[
\theta_i-b_j
\]

以及：

\[
P_{IRT}(Y_{ij}=1)
=
\sigma(\theta_i-b_j)
\]

这是整个项目最重要的模型。

最终重点比较：

\[
\text{XGBoost}
\]

与：

\[
\text{XGBoost + IRT}
\]

---

# 四、不要做的事情

除非核心项目提前全部完成，否则不要做：

- Transformer；
- LSTM；
- Deep Knowledge Tracing；
- 大规模神经网络；
- FastAPI；
- Docker；
- Kubernetes；
- 数据库；
- 复杂 Streamlit 应用；
- 大规模超参数优化；
- MLOps；
- 复杂云部署；
- 追求 SOTA。

不要为了让项目看起来复杂而增加技术栈。

项目目标是：

\[
\boxed{\text{正确 + 清晰 + 完整}}
\]

而不是：

\[
\boxed{\text{庞大 + 复杂}}
\]

---

# 五、数据集

首先选择一个适合完成本项目的 ASSISTments 数据版本。

优先考虑：

- 数据规模适中；
- 有 student ID；
- 有 item/problem ID；
- 有 correctness；
- 有 interaction 顺序或 timestamp；
- 最好有 skill / knowledge component；
- 容易下载；
- 容易复现。

不要为了使用最大数据集增加工作量。

如果 ASSISTments 2009 Skill Builder 或其他经典版本最适合完成该项目，可以直接采用。

在 `README.md` 中用中文记录：

- 数据来源；
- 数据版本；
- 数据规模；
- 主要变量；
- 下载方式；
- 关键数据质量问题；
- 本项目使用了哪些字段。

不要把大型原始数据提交进 GitHub。

---

# 六、预测任务

任务定义为：

根据学生在当前作答之前的历史表现，以及当前题目的信息，预测：

\[
Y_{it}
=
\begin{cases}
1,&\text{作答正确}\\
0,&\text{作答错误}
\end{cases}
\]

输出：

\[
P(Y_{it}=1)
\]

非常重要：

任何用于预测第 \(t\) 次作答的特征，都不能包含第 \(t\) 次作答结果或者未来作答信息。

---

# 七、普通行为特征

不要构建几十个复杂特征。

第一版最多使用大约 5–10 个容易解释的特征。

例如：

## 学生历史作答次数

\[
N_{i,t-1}
\]

## 学生历史正确率

\[
Accuracy_{i,t-1}
=
\frac{
\sum_{s<t}Y_{is}
}{
N_{i,t-1}
}
\]

## 最近若干题正确率

如数据方便，可以加入：

\[
Accuracy^{recent}_{i,t-1}
\]

## skill 历史正确率

如果数据有 skill：

\[
SkillAccuracy_{i,k,t-1}
\]

## skill 历史练习次数

\[
SkillAttempts_{i,k,t-1}
\]

如果存在简单可靠的历史 hint / attempt 信息，可以增加累计历史特征。

但所有特征必须满足：

\[
\boxed{\text{只使用当前时刻以前的信息}}
\]

不要为了增加特征数量而增加复杂度。

---

# 八、数据泄漏

这是本项目必须认真处理的部分。

首先按照：

```text
student → chronological interactions
```

对每个学生按时间或交互顺序排序。

历史正确率必须使用：

```python
shift(1)
```

或等价方式。

例如不能使用：

```text
student overall accuracy
```

因为里面包含当前及未来答案。

应该使用：

```text
student cumulative accuracy before current interaction
```

请在代码中明确实现 causal / historical feature generation。

所有关于数据泄漏的说明，在 README 和 Notebook 中都必须使用中文解释。

---

# 九、数据划分

为了降低项目复杂度，本项目主要研究：

> 对已有一定作答历史的学生预测后续表现。

因此不强制做完全独立的新学生预测。

采用简单的 **within-student temporal split**。

例如每个符合条件的学生：

```text
前 70% interactions → Train
中间 15% → Validation
最后 15% → Test
```

或者根据数据实际情况采用相近比例。

要求：

- Train 一定早于 Validation；
- Validation 一定早于 Test；
- 当前预测不能使用未来信息。

过滤历史太短的学生，例如：

\[
N_i<10
\]

可以删除。

具体阈值根据数据分布决定。

README 中明确用中文说明：

> 本项目评价的是对已有学习历史学生的未来作答预测，而不是 cold-start 新学生泛化。

不要为了同时解决所有预测问题增加项目复杂度。

---

# 十、IRT 部分必须简单

为了保证项目能在一周内完成，优先实现：

\[
\boxed{\text{Rasch / 1PL}}
\]

而不是 2PL、3PL、多维 IRT 或复杂 Bayesian 模型。

模型：

\[
P(Y_{ij}=1)
=
\sigma(\theta_i-b_j)
\]

可以使用成熟的 Python 库。

如果现有库安装、兼容或运行过于麻烦，也可以使用 PyTorch 实现一个简单的 Rasch 模型：

```text
student embedding → theta
item embedding → difficulty
theta - difficulty
↓
sigmoid
↓
probability
```

使用 binary cross entropy 优化。

不要为了“纯手写 IRT”浪费大量时间。

---

# 十一、IRT 必须避免数据泄漏

这一部分必须严格。

只使用：

\[
\boxed{\text{Training interactions}}
\]

拟合 Rasch 模型。

得到：

\[
\hat\theta_i
\]

和：

\[
\hat b_j
\]

然后固定这些参数。

Validation/Test 中不能重新使用未来答案重新拟合 IRT。

为了降低复杂度，本项目允许把：

\[
\hat\theta_i
\]

理解为：

> 根据学生训练阶段历史作答估计出的 baseline ability。

因此测试阶段：

\[
\hat\theta_i
\]

保持固定。

题目难度：

\[
\hat b_j
\]

也必须来自 training data。

如果 test 中存在 training 从未见过的 item，可以：

- 删除这些 interaction；
- 或使用训练集平均 difficulty。

选择最简单且合理的方法，并在 README 中用中文说明。

---

# 十二、IRT 特征

给每个 Validation/Test interaction 加入：

## Ability

\[
\hat\theta_i
\]

## Item difficulty

\[
\hat b_j
\]

## Ability–difficulty gap

\[
Gap_{ij}
=
\hat\theta_i-\hat b_j
\]

## IRT probability

\[
P_{IRT}
=
\sigma(\hat\theta_i-\hat b_j)
\]

因此最终形成两组特征。

### Behavioral features

```text
historical_accuracy
historical_attempts
skill_accuracy
skill_attempts
...
```

### Behavioral + Psychometric features

```text
historical_accuracy
historical_attempts
skill_accuracy
skill_attempts
theta
item_difficulty
theta_minus_difficulty
irt_probability
```

---

# 十三、模型训练

只做：

## Logistic Regression

用于 baseline。

## XGBoost

作为主要机器学习模型。

不要进行复杂的 hyperparameter tuning。

只做少量合理参数设置，例如：

- max_depth；
- learning_rate；
- n_estimators；
- subsample。

可以使用一个很小的手动参数组合或 RandomizedSearch。

不要为了找到最优 0.001 AUC 浪费时间。

---

# 十四、模型比较

最终至少得到：

| 模型 | AUC | LogLoss | Brier Score |
|---|---:|---:|---:|
| Logistic Regression | | | |
| Rasch IRT | | | |
| XGBoost | | | |
| XGBoost + IRT | | | |

主要指标：

\[
AUC
\]

同时报告：

\[
LogLoss
\]

和：

\[
Brier\ Score
\]

因为这是概率预测任务。

如果实现很容易，也可以额外报告：

- Accuracy；
- F1。

但它们不是重点。

所有模型结果解释必须使用中文。

---

# 十五、核心分析

项目最重要的一张结果表就是：

\[
\boxed{
XGBoost
\quad vs.\quad
XGBoost+IRT
}
\]

计算：

\[
\Delta AUC
=
AUC_{Hybrid}-AUC_{XGB}
\]

\[
\Delta Brier
=
Brier_{Hybrid}-Brier_{XGB}
\]

不要假设加入 IRT 一定会提升性能。

如果没有提升，也要如实报告。

在 README 和最终总结中用中文讨论：

> 行为历史特征是否已经包含大量与 ability 相同的信息；

> IRT 特征是否提供了额外预测价值；

> 如果提升有限，可能说明什么；

> 如果明显提升，IRT 可能补充了哪些信息。

---

# 十六、简单可解释性

不要进行复杂 XAI。

只需要：

## XGBoost feature importance

最好增加：

## SHAP summary plot

如果 SHAP 安装和使用顺利。

重点观察：

- historical accuracy；
- \(\theta\)；
- item difficulty；
- \(\theta-b\)；
- IRT probability；

是否成为重要预测变量。

如果 SHAP 增加明显技术麻烦，可以只完成 feature importance。

图表标题、坐标轴、图注优先使用中文。

---

# 十七、简单可视化

至少生成以下图：

## Figure 1
学生交互次数分布。

## Figure 2
总体作答正确率或正确/错误比例。

## Figure 3
IRT 学生能力参数 \(\theta\) 分布。

## Figure 4
IRT 题目难度参数 \(b\) 分布。

## Figure 5
模型性能比较。

## Figure 6（可选）
XGBoost + IRT feature importance / SHAP。

图表标题、坐标轴标签、图注优先使用中文。

不要为了做很多图增加工作量。

---

# 十八、项目目录保持简单

不要建立大型企业项目结构。

建议：

```text
assistments-irt-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── irt.py
│   ├── train.py
│   └── evaluate.py
│
├── artifacts/
│   ├── results.csv
│   └── figures/
│
├── resume_bullets.md
├── interview_notes.md
│
└── tests/
    └── test_core.py
```

如果还能进一步简化，可以合理调整。

但不要把所有正式逻辑全部放进 Notebook。

---

# 十九、测试保持简单

不需要完整企业级 test suite。

只需要几个最重要的 sanity checks。

## Test 1

历史正确率不包含当前 response。

## Test 2

时间顺序正确：

\[
Train < Validation < Test
\]

## Test 3

IRT：

固定 difficulty 时：

\[
\theta\uparrow
\Rightarrow
P(correct)\uparrow
\]

固定 ability 时：

\[
b\uparrow
\Rightarrow
P(correct)\downarrow
\]

这些测试足以体现基本工程严谨性。

测试说明与测试结果总结使用中文。

---

# 二十、README.md 必须全部使用中文

`README.md` 是项目的重要成果，必须全部使用中文撰写。

不要自动生成英文 README。

推荐结构如下：

# ASSISTments + IRT 学生作答预测

一句中文项目简介，例如：

> 基于 ASSISTments 学习行为数据，结合 Rasch IRT 与 XGBoost，对学生后续作答正确率进行预测，并评估心理测量特征是否能够提升机器学习模型表现。

## 1. 项目简介

用中文说明：

- 项目做什么；
- 为什么选择教育预测；
- 为什么加入 IRT；
- 这个项目希望展示哪些能力。

## 2. 数据集

用中文说明：

- ASSISTments 数据版本；
- 数据来源；
- 样本规模；
- 主要字段；
- 下载方式；
- 使用限制。

## 3. 预测任务

解释：

\[
P(correct)
\]

对应什么实际问题。

## 4. 数据处理

用中文说明：

- 数据清洗；
- 学生内排序；
- 历史特征构造；
- temporal split；
- 删除规则。

## 5. 数据泄漏控制

重点用中文解释：

- 为什么不能使用总体正确率；
- 为什么使用 shift；
- 为什么 IRT 只能在训练数据拟合；
- 为什么 Validation/Test 不能更新能力参数。

## 6. 模型

用中文介绍：

- Logistic Regression；
- Rasch IRT；
- XGBoost；
- XGBoost + IRT。

## 7. 心理测量特征

解释：

\[
P(Y_{ij}=1)=\sigma(\theta_i-b_j)
\]

并用中文说明：

- \(\theta\)：学生能力；
- \(b\)：题目难度；
- \(\theta-b\)：能力与题目难度匹配程度；
- \(P_{IRT}\)：IRT 预测正确概率。

## 8. 结果

放真实结果表。

例如：

| 模型 | AUC | LogLoss | Brier Score |
|---|---:|---:|---:|
| Logistic Regression | | | |
| Rasch IRT | | | |
| XGBoost | | | |
| XGBoost + IRT | | | |

## 9. 主要发现

用中文总结 3–5 条真实发现。

不要提前假定加入 IRT 一定提升模型。

## 10. 可解释性分析

说明 IRT 特征和行为特征的重要性。

## 11. 项目结构

解释主要文件作用。

## 12. 如何运行

使用中文说明具体命令。

## 13. 项目局限

至少讨论：

- 主要预测已有学生；
- 使用 Rasch/1PL 而非复杂 IRT；
- \(\theta\) 在测试阶段固定；
- 没有建立动态知识状态模型；
- ASSISTments 数据本身的限制。

## 14. 后续可以扩展的方向

只简单列出：

- 2PL；
- 动态 IRT；
- knowledge tracing；
- LSTM/Transformer；
- 更严格的新学生泛化。

不要真的在本项目中继续实现。

---

# 二十一、resume_bullets.md 必须使用中文

项目完成后创建：

```text
resume_bullets.md
```

该文件必须使用中文。

目标是生成可以直接放到中文简历中的项目经历描述。

格式建议为：

## 项目名称

**ASSISTments + IRT 学生作答预测**

## 项目简介

用 1–2 句话概括项目。

## 简历要点

生成 3–4 条简历 bullet，重点体现：

- 数据处理；
- 数据泄漏控制；
- XGBoost；
- Rasch IRT；
- 心理测量特征融合；
- AUC / LogLoss / Brier Score；
- 实际结果。

例如：

- 基于 ASSISTments 学习行为数据构建学生后续作答预测流程，完成学生内时间排序、历史行为特征构造与 temporal split，避免未来信息泄漏。
- 使用 Rasch IRT 估计学生能力与题目难度，并将 \(\theta\)、\(b\)、\(\theta-b\) 与 IRT 预测概率作为心理测量特征融入 XGBoost。
- 对比 Logistic Regression、Rasch IRT、XGBoost 与 XGBoost + IRT，使用 AUC、LogLoss 和 Brier Score 综合评价概率预测性能。
- 如果最终结果存在真实提升，则写入真实提升幅度；如果没有提升，不得虚构。

简历 bullet 要简洁、技术导向，不要写成论文摘要。

不要输出英文简历版本，除非我之后明确要求。

---

# 二十二、interview_notes.md 必须使用中文

项目完成后创建：

```text
interview_notes.md
```

该文件全部使用中文。

用于帮助我准备求职面试。

至少回答以下问题：

1. 这个项目解决什么问题？
2. 为什么选择 ASSISTments？
3. 为什么使用 XGBoost？
4. 为什么加入 IRT？
5. Rasch 模型是什么？
6. \(\theta\) 代表什么？
7. \(b\) 代表什么？
8. 为什么不能直接使用学生总体正确率？
9. 什么是数据泄漏？
10. 为什么采用 temporal split？
11. 为什么 IRT 只能在 training data 上拟合？
12. 为什么 Validation/Test 阶段不能用未来答案更新 \(\theta\)？
13. IRT 特征为什么可能帮助 XGBoost？
14. 如果 IRT 没有提升 AUC，怎么解释？
15. 为什么不能只看 Accuracy？
16. AUC、LogLoss 和 Brier Score 分别反映什么？
17. XGBoost 与 Logistic Regression 的差异是什么？
18. 本项目最重要的技术难点是什么？
19. 本项目有什么局限？
20. 如果再给一周时间，你会增加什么？

答案不要写成教科书定义。

应该结合本项目真实实现，用适合技术面试口头表达的中文回答。

---

# 二十三、Notebook 也必须中文化

`notebooks/01_eda.ipynb` 中：

- Markdown 标题使用中文；
- 分析说明使用中文；
- 图表解释使用中文；
- 结果小结使用中文。

代码变量本身可以继续使用英文。

例如：

```python
historical_accuracy
item_difficulty
irt_probability
```

不要求翻译成中文变量名。

---

# 二十四、代码注释与文档字符串

代码本身保持常见 Python 英文命名规范。

但：

- 关键代码注释优先用中文；
- docstring 优先用中文；
- 复杂逻辑必须有中文解释；
- 尤其是 temporal feature、IRT fitting 和 leakage prevention 部分必须注释清楚。

不要因为追求“专业感”而强行全部使用英文注释。

---

# 二十五、执行 Phase

不要一次生成整个项目。

请按照下面 Phase 持续推进。

## Phase 0 — 范围与数据

目标：

- 检查项目目录；
- 检查环境；
- 确定 ASSISTments 版本；
- 下载或明确下载方式；
- 确认字段；
- 建立项目结构。

---

## Phase 1 — 数据与特征

完成：

- 数据清洗；
- 学生内时间排序；
- temporal split；
- causal historical features；
- 简单 EDA；
- leakage sanity check。

完成后必须实际运行确认。

---

## Phase 2 — Rasch IRT

完成：

\[
P(Y_{ij}=1)
=
\sigma(\theta_i-b_j)
\]

只使用 training interactions 拟合。

输出：

- student ability；
- item difficulty；
- IRT probability。

完成基本 sanity check。

---

## Phase 3 — 机器学习

完成：

- Logistic Regression；
- XGBoost；
- XGBoost + IRT。

不要复杂调参。

---

## Phase 4 — 评价

计算：

- AUC；
- LogLoss；
- Brier Score。

生成：

- 模型结果表；
- 核心图；
- feature importance / SHAP（如果容易实现）。

所有解释使用中文。

---

## Phase 5 — 求职作品整理

完成：

- 中文 `README.md`；
- `requirements.txt`；
- clean project structure；
- `results.csv`；
- 中文图表；
- 中文 `resume_bullets.md`；
- 中文 `interview_notes.md`。

到这里项目即视为完成。

---

# 二十六、每个 Phase 的工作方式

开始 Phase 时使用中文简要告诉我：

```text
Phase X — 名称

目标：
...

准备创建或修改的文件：
...

主要技术决策：
...
```

然后直接执行代码和文件修改。

完成以后使用中文报告：

```text
已完成：
...

结果：
...

检查：
...

发现的问题：
...

下一步：
...
```

然后继续下一 Phase。

除非遇到真正无法自行解决的外部问题，否则不要反复询问我。

合理的小型技术选择由你自行决定。

---

# 二十七、时间优先级

整个项目的优先级严格按照：

\[
\boxed{
\text{完成项目}
>
\text{代码正确}
>
\text{IRT 特色}
>
\text{结果清晰}
>
\text{模型复杂度}
}
\]

如果某一项工作明显会让项目超出一周，请主动简化。

例如：

2PL 太复杂：

\[
\rightarrow 1PL
\]

SHAP 出问题：

\[
\rightarrow feature\ importance
\]

完整数据训练太慢：

\[
\rightarrow 合理抽样
\]

复杂超参数搜索太慢：

\[
\rightarrow 固定合理参数
\]

不要让非核心功能阻止整个项目完成。

---

# 二十八、最低完成标准

达到以下条件就可以结束项目：

- [ ] ASSISTments 数据可获取
- [ ] causal 历史特征完成
- [ ] temporal split 完成
- [ ] 无明显 future leakage
- [ ] Logistic Regression 完成
- [ ] Rasch IRT 完成
- [ ] XGBoost 完成
- [ ] XGBoost + IRT 完成
- [ ] AUC 完成
- [ ] LogLoss 完成
- [ ] Brier Score 完成
- [ ] 一张完整模型比较表
- [ ] 3–6 张有用图
- [ ] 中文 README 完成
- [ ] requirements 完成
- [ ] 中文 resume_bullets.md 完成
- [ ] 中文 interview_notes.md 完成
- [ ] Notebook 主要说明已中文化

完成这些以后，不要自行扩展大型新功能。

---

# 二十九、现在开始

现在执行：

## Phase 0 — 范围与数据

首先：

1. 检查当前目录和环境；
2. 确定最合适、最容易完成项目的 ASSISTments 数据版本；
3. 检查该版本具体字段；
4. 判断是否满足 student、item、correctness 和 interaction order 要求；
5. 建立最简单合理的项目目录；
6. 给出最终 MVP 技术方案。

然后直接推进 Phase 1。

记住：

这是一个**一周内完成的中文求职作品集项目**，不是论文，不是 SOTA 竞赛，也不是生产级系统。

最重要的是把：

\[
\boxed{
ASSISTments
+
Rasch\ IRT
+
XGBoost
+
Leakage\ Prevention
}
\]

这条主线真正做完整。

并且：

\[
\boxed{\text{所有对外展示文档默认使用中文}}
\]

尤其是：

- `README.md`
- `resume_bullets.md`
- `interview_notes.md`
- Notebook Markdown
- 项目结果总结
- 图表标题与说明
