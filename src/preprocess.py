"""数据读取、清洗和学生内时间切分。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"user_id", "problem_id", "correct", "order_id"}
DEFAULT_MIN_INTERACTIONS = 10


def _read_csv_with_fallback(path: str | Path) -> pd.DataFrame:
    """读取 CSV；优先使用 UTF-8，失败时尝试常见中文兼容编码。"""
    path = Path(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)


def load_and_clean(path: str | Path, min_interactions: int = DEFAULT_MIN_INTERACTIONS) -> pd.DataFrame:
    """读取 ASSISTments 修正版并保留本项目所需字段。"""
    df = _read_csv_with_fallback(path)
    df.columns = [str(column).strip() for column in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"数据缺少必需字段: {sorted(missing)}；实际字段为: {list(df.columns)}")

    if "skill_id" in df.columns:
        skill_column = "skill_id"
    elif "skill_name" in df.columns:
        skill_column = "skill_name"
    else:
        skill_column = None

    columns = ["user_id", "problem_id", "correct", "order_id"]
    if skill_column:
        columns.append(skill_column)
    clean = df[columns].copy()
    clean = clean.rename(columns={skill_column: "skill_id"} if skill_column else {})
    if "skill_id" not in clean.columns:
        clean["skill_id"] = "unknown_skill"

    # 先删除关键字段缺失，再统一为稳定的字符串 ID 和数值标签。
    clean = clean.dropna(subset=["user_id", "problem_id", "correct", "order_id"])
    clean["correct"] = pd.to_numeric(clean["correct"], errors="coerce")
    clean["order_id"] = pd.to_numeric(clean["order_id"], errors="coerce")
    clean = clean.dropna(subset=["correct", "order_id"])
    clean = clean[clean["correct"].isin([0, 1])].copy()
    clean["correct"] = clean["correct"].astype(int)
    clean["user_id"] = clean["user_id"].astype(str)
    clean["problem_id"] = clean["problem_id"].astype(str)
    clean["skill_id"] = clean["skill_id"].fillna("unknown_skill").astype(str)
    clean["_original_row"] = np.arange(len(clean))

    clean = clean.sort_values(["user_id", "order_id", "_original_row"], kind="mergesort")
    # 官方修正版已将多技能题合并为一行；这里再次去重以防下载文件仍含重复记录。
    clean = clean.drop_duplicates(["user_id", "order_id", "problem_id"], keep="first")
    clean = clean.drop(columns="_original_row")

    counts = clean.groupby("user_id")["problem_id"].transform("size")
    clean = clean[counts >= min_interactions].copy()
    if clean.empty:
        raise ValueError("清洗后没有满足最小历史长度要求的学生，请检查数据文件或阈值。")

    clean = clean.sort_values(["user_id", "order_id"], kind="mergesort").reset_index(drop=True)
    return clean


def assign_temporal_split(df: pd.DataFrame, train_ratio: float = 0.70, validation_ratio: float = 0.15) -> pd.DataFrame:
    """为每名学生按时间顺序标记 train、validation、test。"""
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("切分比例必须满足 0 < train_ratio < 1 且 validation_ratio >= 0。")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("训练集和验证集比例之和必须小于 1。")

    result = df.sort_values(["user_id", "order_id"], kind="mergesort").copy()
    result["split"] = "train"
    for user_id, indices in result.groupby("user_id", sort=False).groups.items():
        ordered_indices = list(indices)
        n = len(ordered_indices)
        n_train = max(1, int(np.floor(n * train_ratio)))
        n_validation = max(1, int(np.floor(n * validation_ratio)))
        # 保证 Test 至少有一条记录；数据已过滤为 n >= 10，通常不会触发调整。
        if n_train + n_validation >= n:
            n_validation = max(1, n - n_train - 1)
        train_end = n_train
        validation_end = n_train + n_validation
        result.loc[ordered_indices[train_end:validation_end], "split"] = "validation"
        result.loc[ordered_indices[validation_end:], "split"] = "test"
    return result.reset_index(drop=True)


def prepare_dataset(input_path: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
    """完成清洗与时间切分，并可选保存处理后的交互表。"""
    result = assign_temporal_split(load_and_clean(input_path))
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗 ASSISTments 数据并生成学生内时间切分。")
    parser.add_argument("--input", required=True, help="原始 ASSISTments CSV 路径")
    parser.add_argument("--output", default="data/processed/interactions.csv", help="处理后 CSV 输出路径")
    args = parser.parse_args()
    result = prepare_dataset(args.input, args.output)
    print(f"已保存: {args.output}")
    print(f"交互数: {len(result):,}；学生数: {result['user_id'].nunique():,}")
    print(result["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
