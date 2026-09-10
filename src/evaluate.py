"""指标计算和中文图表输出。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score


_chinese_font = None
for _font_path in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"]:
    if Path(_font_path).exists():
        font_manager.fontManager.addfont(_font_path)
        _chinese_font = font_manager.FontProperties(fname=_font_path).get_name()
        break
plt.rcParams["font.sans-serif"] = [_chinese_font or "DejaVu Sans", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font=_chinese_font or "DejaVu Sans")


def compute_metrics(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """计算概率预测指标和阈值 0.5 下的辅助分类指标。"""
    y_true = np.asarray(y_true).astype(int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    predicted = (probability >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, probability)),
        "logloss": float(log_loss(y_true, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_eda_figures(df: pd.DataFrame, irt_features: pd.DataFrame, output_dir: str | Path) -> None:
    """保存交互次数、正确率、能力和难度分布图。"""
    output_dir = Path(output_dir)
    counts = df.groupby("user_id").size()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(counts, bins=30, ax=ax, color="#4C78A8")
    ax.set_title("学生交互次数分布")
    ax.set_xlabel("每名学生的交互次数")
    ax.set_ylabel("学生人数")
    _save(fig, output_dir / "interaction_count_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    correctness = df["correct"].value_counts().reindex([0, 1], fill_value=0)
    ax.bar(["错误", "正确"], correctness.values, color=["#E45756", "#54A24B"])
    ax.set_title("总体作答正确与错误分布")
    ax.set_xlabel("作答结果")
    ax.set_ylabel("交互次数")
    _save(fig, output_dir / "correctness_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(irt_features["theta"], bins=30, ax=ax, color="#F58518")
    ax.set_title("Rasch 学生能力参数分布")
    ax.set_xlabel("学生能力 θ")
    ax.set_ylabel("交互次数")
    _save(fig, output_dir / "theta_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    item_values = irt_features.groupby(df["problem_id"].astype(str))["item_difficulty"].first()
    sns.histplot(item_values, bins=30, ax=ax, color="#72B7B2")
    ax.set_title("Rasch 题目难度参数分布")
    ax.set_xlabel("题目难度 b")
    ax.set_ylabel("题目数量")
    _save(fig, output_dir / "item_difficulty_distribution.png")


def save_model_comparison(results: pd.DataFrame, output_path: str | Path) -> None:
    """保存 Test 集三项核心指标比较图。"""
    test_results = results[results["split"] == "test"].copy()
    order = ["Logistic Regression", "Rasch IRT", "XGBoost", "XGBoost + IRT"]
    test_results["model"] = pd.Categorical(test_results["model"], categories=order, ordered=True)
    test_results = test_results.sort_values("model")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(
        axes,
        ["auc", "logloss", "brier_score"],
        ["AUC（越高越好）", "LogLoss（越低越好）", "Brier Score（越低越好）"],
    ):
        ax.bar(test_results["model"].astype(str), test_results[metric], color="#4C78A8")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel({"auc": "AUC", "logloss": "LogLoss", "brier_score": "Brier Score"}[metric])
    _save(fig, output_path)


def save_feature_importance(model, feature_names: list[str], output_path: str | Path) -> None:
    """保存 XGBoost 原生特征重要性。"""
    values = np.asarray(model.feature_importances_, dtype=float)
    importance = pd.DataFrame({"feature": feature_names, "importance": values}).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance["feature"], importance["importance"], color="#ECA82C")
    ax.set_title("XGBoost + IRT 特征重要性")
    ax.set_xlabel("重要性")
    ax.set_ylabel("特征")
    _save(fig, output_path)
