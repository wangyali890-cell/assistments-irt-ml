# 项目名称

**ASSISTments + IRT 学生作答预测**

## 项目简介

基于 ASSISTments 2009–2010 Skill Builder 学习行为数据，构建学生后续作答正确率预测流程，并将 Rasch IRT 估计的学生能力与题目难度融合到 XGBoost 中。

## 简历要点

- 基于 34.2 万条 ASSISTments 学习交互完成数据清洗、学生内时间排序、70%/15%/15% temporal split 和 causal 历史特征构造，避免当前答案及未来信息泄漏。
- 使用训练阶段数据拟合 Rasch/1PL IRT，估计学生能力 (	heta) 与题目难度 (b)，并将 (	heta)、(b)、(	heta-b) 和 IRT 正确概率融合进 XGBoost。
- 对比 Logistic Regression、Rasch IRT、XGBoost 与 XGBoost + IRT，使用 AUC、LogLoss 和 Brier Score 评价概率预测性能。
- 在 Test 集上，XGBoost + IRT 的 AUC 达到 0.7667，较普通 XGBoost 提升 0.0486；Brier Score 从 0.1966 降至 0.1893。
