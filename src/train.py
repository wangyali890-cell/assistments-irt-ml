"""训练四个 MVP 模型并生成结果、图表和处理数据。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .evaluate import (
    compute_metrics,
    save_eda_figures,
    save_feature_importance,
    save_model_comparison,
)
from .features import BEHAVIOR_FEATURES, IRT_FEATURES, add_causal_features, get_behavior_matrix, get_hybrid_matrix
from .irt import RaschIRT
from .preprocess import prepare_dataset


def build_xgb() -> XGBClassifier:
    """构建固定参数的轻量 XGBoost。"""
    return XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=2,
    )


def train_and_evaluate(input_path: str | Path, processed_path: str | Path, results_path: str | Path, figure_dir: str | Path) -> pd.DataFrame:
    """执行完整训练流程；所有模型只在 Train 数据上拟合。"""
    interactions = prepare_dataset(input_path)
    data = add_causal_features(interactions)

    train = data[data["split"] == "train"].copy()
    validation = data[data["split"] == "validation"].copy()
    test = data[data["split"] == "test"].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Train、Validation、Test 不能有空集合。")

    # Rasch 参数只来自训练阶段；之后对全体行做固定映射。
    irt = RaschIRT(C=1.0, max_iter=300).fit(train)
    irt_features = irt.transform(data).reset_index(drop=True)
    data = data.reset_index(drop=True)
    data = pd.concat([data, irt_features], axis=1)
    train = data[data["split"] == "train"].copy()
    validation = data[data["split"] == "validation"].copy()
    test = data[data["split"] == "test"].copy()

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed_path, index=False, encoding="utf-8-sig")

    X_train_behavior = get_behavior_matrix(train)
    X_validation_behavior = get_behavior_matrix(validation)
    X_test_behavior = get_behavior_matrix(test)
    X_train_hybrid = get_hybrid_matrix(train)
    X_validation_hybrid = get_hybrid_matrix(validation)
    X_test_hybrid = get_hybrid_matrix(test)
    y_train = train["correct"].astype(int)

    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=42)),
        ]
    )
    logistic.fit(X_train_behavior, y_train)

    xgb_behavior = build_xgb().fit(X_train_behavior, y_train)
    xgb_hybrid = build_xgb().fit(X_train_hybrid, y_train)

    predictions = {
        "Logistic Regression": (
            logistic.predict_proba(X_validation_behavior)[:, 1],
            logistic.predict_proba(X_test_behavior)[:, 1],
        ),
        "Rasch IRT": (
            validation["irt_probability"].to_numpy(),
            test["irt_probability"].to_numpy(),
        ),
        "XGBoost": (
            xgb_behavior.predict_proba(X_validation_behavior)[:, 1],
            xgb_behavior.predict_proba(X_test_behavior)[:, 1],
        ),
        "XGBoost + IRT": (
            xgb_hybrid.predict_proba(X_validation_hybrid)[:, 1],
            xgb_hybrid.predict_proba(X_test_hybrid)[:, 1],
        ),
    }

    rows: list[dict[str, object]] = []
    for model_name, (validation_probability, test_probability) in predictions.items():
        for split_name, frame, probability in [
            ("validation", validation, validation_probability),
            ("test", test, test_probability),
        ]:
            metrics = compute_metrics(frame["correct"], probability)
            rows.append({"model": model_name, "split": split_name, **metrics})

    results = pd.DataFrame(rows)
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False, encoding="utf-8-sig")

    figure_dir = Path(figure_dir)
    save_eda_figures(data, data[["theta", "item_difficulty"]], figure_dir)
    save_model_comparison(results, figure_dir / "model_comparison.png")
    save_feature_importance(xgb_hybrid, BEHAVIOR_FEATURES + IRT_FEATURES, figure_dir / "feature_importance.png")

    print(f"已保存处理数据: {processed_path}")
    print(f"已保存结果表: {results_path}")
    print(f"已保存图表目录: {figure_dir}")
    print("\nTest 集结果:")
    print(results[results["split"] == "test"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 ASSISTments + Rasch IRT 学生作答预测模型。")
    parser.add_argument("--input", default="data/raw/skill_builder_data_corrected.csv")
    parser.add_argument("--processed", default="data/processed/features.csv")
    parser.add_argument("--results", default="artifacts/results.csv")
    parser.add_argument("--figures", default="artifacts/figures")
    args = parser.parse_args()
    train_and_evaluate(args.input, args.processed, args.results, args.figures)


if __name__ == "__main__":
    main()
