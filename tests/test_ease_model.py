"""Tests for module.py"""

import sys

sys.path.append("src")

from fastEASE import PipelineEASE


def test_ease_class() -> None:
    # Arrange
    dummy = PipelineEASE(
        user_item_it=zip(map(str, range(10)), map(str, range(10))),
        min_item_freq=0,
        min_user_interactions_len=0,
        max_user_interactions_len=11,
    )
    metrics = dummy.calc_ndcg_at_k()
    assert isinstance(metrics, dict)
