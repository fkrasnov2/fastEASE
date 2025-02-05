"""Tests for module.py"""

from src.fastEASE import PipelineEASE


def test_ease_class() -> None:
    # Arrange
    dummy = PipelineEASE(
        user_item_it=zip(map(str, range(10)), map(str, range(10))),
        min_item_freq=0,
        min_user_interactions_len=0,
        max_user_interactions_len=11,
        calc_ndcg_at_k=True,
        k=2,
        prediction_batch_size=2,
        predict_next_n=False,
        next_n=3,
        regularization=100,
    )
    print(dummy.ndcg)
    # Assert
    assert dummy
