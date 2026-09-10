"""项目核心 sanity checks。"""

import numpy as np
import pandas as pd

from src.features import add_causal_features
from src.irt import RaschIRT
from src.preprocess import assign_temporal_split


def tiny_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["s1"] * 6 + ["s2"] * 6,
            "problem_id": ["p1", "p2", "p1", "p3", "p2", "p4"] * 2,
            "skill_id": ["k1", "k1", "k1", "k2", "k1", "k2"] * 2,
            "correct": [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1],
            "order_id": list(range(1, 7)) * 2,
        }
    )


def test_historical_accuracy_excludes_current_response() -> None:
    result = add_causal_features(tiny_interactions())
    first_student = result[result["user_id"] == "s1"]
    assert first_student.iloc[0]["student_attempts_before"] == 0
    assert first_student.iloc[0]["student_historical_accuracy"] == 0.5
    assert first_student.iloc[1]["student_attempts_before"] == 1
    assert first_student.iloc[1]["student_historical_accuracy"] == 1.0
    assert first_student.iloc[2]["student_historical_accuracy"] == 0.5


def test_temporal_split_is_ordered_within_student() -> None:
    result = assign_temporal_split(tiny_interactions(), train_ratio=0.5, validation_ratio=0.25)
    for _, group in result.groupby("user_id"):
        split_order = group.sort_values("order_id")["split"].tolist()
        assert split_order == sorted(split_order, key={"train": 0, "validation": 1, "test": 2}.get)


def test_rasch_probability_monotonicity() -> None:
    assert RaschIRT.probability(1.0, 0.0) > RaschIRT.probability(0.0, 0.0)
    assert RaschIRT.probability(0.0, 1.0) < RaschIRT.probability(0.0, 0.0)


def test_unseen_item_uses_training_mean_difficulty() -> None:
    train = tiny_interactions().iloc[:8].copy()
    model = RaschIRT(max_iter=100).fit(train)
    new = pd.DataFrame({"user_id": ["s1"], "problem_id": ["unseen"], "correct": [1]})
    transformed = model.transform(new)
    assert np.isclose(transformed.iloc[0]["item_difficulty"], model.default_difficulty_)
