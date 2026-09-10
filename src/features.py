"""只使用当前交互之前信息的因果历史特征。"""

from __future__ import annotations

import pandas as pd


BEHAVIOR_FEATURES = [
    "student_attempts_before",
    "student_historical_accuracy",
    "recent_accuracy_5",
    "skill_attempts_before",
    "skill_historical_accuracy",
]

IRT_FEATURES = [
    "theta",
    "item_difficulty",
    "theta_minus_difficulty",
    "irt_probability",
]


def add_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """生成学生和 skill 的历史特征，所有累计量均排除当前答案。"""
    required = {"user_id", "skill_id", "correct", "order_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"生成历史特征时缺少字段: {sorted(missing)}")

    result = df.sort_values(["user_id", "order_id"], kind="mergesort").reset_index(drop=True).copy()

    student_group = result.groupby("user_id", sort=False)
    result["student_attempts_before"] = student_group.cumcount()
    result["student_correct_before"] = student_group["correct"].cumsum() - result["correct"]
    student_denominator = result["student_attempts_before"].replace(0, float("nan"))
    result["student_historical_accuracy"] = (
        result["student_correct_before"] / student_denominator
    ).astype(float).fillna(0.5)

    # shift(1) 明确保证最近正确率不包含当前 response。
    result["recent_accuracy_5"] = (
        result.groupby("user_id", sort=False)["correct"]
        .transform(lambda values: values.shift(1).rolling(5, min_periods=1).mean())
        .astype(float)
        .fillna(0.5)
    )

    skill_group = result.groupby(["user_id", "skill_id"], sort=False)
    result["skill_attempts_before"] = skill_group.cumcount()
    result["skill_correct_before"] = skill_group["correct"].cumsum() - result["correct"]
    skill_denominator = result["skill_attempts_before"].replace(0, float("nan"))
    result["skill_historical_accuracy"] = (
        result["skill_correct_before"] / skill_denominator
    ).astype(float).fillna(0.5)

    return result


def get_behavior_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """返回供普通机器学习模型使用的行为特征矩阵。"""
    missing = set(BEHAVIOR_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"缺少行为特征: {sorted(missing)}")
    return df[BEHAVIOR_FEATURES].astype(float)


def get_hybrid_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """返回行为特征与 IRT 特征的组合矩阵。"""
    missing = set(BEHAVIOR_FEATURES + IRT_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"缺少混合模型特征: {sorted(missing)}")
    return df[BEHAVIOR_FEATURES + IRT_FEATURES].astype(float)
