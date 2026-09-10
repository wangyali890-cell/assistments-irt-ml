"""使用稀疏逻辑回归实现简单 Rasch / 1PL IRT。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression


@dataclass
class RaschIRT:
    """Rasch 模型：logit(P(correct)) = theta_student - difficulty_item。"""

    C: float = 1.0
    max_iter: int = 300
    random_state: int = 42

    def fit(self, df: pd.DataFrame) -> "RaschIRT":
        required = {"user_id", "problem_id", "correct"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"拟合 Rasch 时缺少字段: {sorted(missing)}")
        if df.empty:
            raise ValueError("不能在空数据上拟合 Rasch 模型。")

        users = df["user_id"].astype(str)
        items = df["problem_id"].astype(str)
        user_codes, user_values = pd.factorize(users, sort=True)
        item_codes, item_values = pd.factorize(items, sort=True)
        n_users = len(user_values)
        n_items = len(item_values)
        n_rows = len(df)

        row_ids = np.arange(n_rows)
        columns = np.concatenate([user_codes, n_users + item_codes])
        rows = np.concatenate([row_ids, row_ids])
        values = np.concatenate([np.ones(n_rows), -np.ones(n_rows)])
        design = coo_matrix(
            (values, (rows, columns)),
            shape=(n_rows, n_users + n_items),
        ).tocsr()

        self.model_ = LogisticRegression(
            C=self.C,
            solver="liblinear",
            fit_intercept=False,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model_.fit(design, df["correct"].astype(int).to_numpy())

        coefficients = self.model_.coef_[0]
        raw_theta = coefficients[:n_users]
        # 题目列在设计矩阵中已经乘以 -1，因此对应系数本身就是 b。
        raw_difficulty = coefficients[n_users:]

        # theta 与 b 同时加减同一常数不会改变 theta-b；令平均题目难度为 0，便于解释。
        shift = float(raw_difficulty.mean())
        self.theta_ = raw_theta - shift
        self.difficulty_ = raw_difficulty - shift
        self.user_to_theta_ = dict(zip(user_values.astype(str), self.theta_))
        self.item_to_difficulty_ = dict(zip(item_values.astype(str), self.difficulty_))
        self.default_theta_ = float(np.mean(self.theta_))
        self.default_difficulty_ = float(np.mean(self.difficulty_))
        self.n_train_interactions_ = n_rows
        return self

    @staticmethod
    def probability(theta: float | np.ndarray, difficulty: float | np.ndarray) -> np.ndarray:
        """计算 Rasch 正确概率。"""
        return expit(np.asarray(theta) - np.asarray(difficulty))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """将固定的学生能力和题目难度映射到新数据。"""
        if not hasattr(self, "user_to_theta_"):
            raise RuntimeError("请先调用 fit。")
        users = df["user_id"].astype(str)
        items = df["problem_id"].astype(str)
        theta = users.map(self.user_to_theta_).fillna(self.default_theta_).astype(float)
        difficulty = items.map(self.item_to_difficulty_).fillna(self.default_difficulty_).astype(float)
        output = pd.DataFrame(index=df.index)
        output["theta"] = theta
        output["item_difficulty"] = difficulty
        output["theta_minus_difficulty"] = theta - difficulty
        output["irt_probability"] = self.probability(theta.to_numpy(), difficulty.to_numpy())
        return output

    def fit_transform(self, train_df: pd.DataFrame, all_df: pd.DataFrame) -> pd.DataFrame:
        """只用 train_df 拟合，并为 all_df 生成固定 IRT 特征。"""
        return self.fit(train_df).transform(all_df)
