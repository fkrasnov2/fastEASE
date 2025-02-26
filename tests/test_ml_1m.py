import sys

sys.path.append("src")

from collections.abc import Iterable

import pytest

from fastEASE import PipelineEASE


class PipelineML1M(PipelineEASE):
    def __init__(self, path_to_dataset: str, **kwargs):
        kwargs.update({"user_item_it": self.load_interactions(path_to_dataset)})
        super().__init__(**kwargs)

    @staticmethod
    def load_interactions(path_to_dataset) -> Iterable[tuple[int, int]]:
        path_to_interactions = path_to_dataset + "/" + "ratings.dat"
        with open(path_to_interactions, "r") as file:
            for line in file:
                yield tuple(map(int, line.strip("\n").split("::")[:2]))


@pytest.fixture
def pipeline_ml_1m():
    return PipelineML1M("dataset/ml-1m")


def test_items_vocab(pipeline_ml_1m):
    assert len(pipeline_ml_1m.items_vocab) > 1000


def test_users_vocab(pipeline_ml_1m):
    assert len(pipeline_ml_1m.users_vocab) > 500


def test_interactions_matrix(pipeline_ml_1m):
    assert pipeline_ml_1m.interactions_matrix.shape == (
        len(pipeline_ml_1m.users_vocab),
        len(pipeline_ml_1m.items_vocab),
    )


def test_ndcg():
    pipeline = PipelineML1M(
        "dataset/ml-1m",
        min_item_freq=1,
        min_user_interactions_len=5,
        max_user_interactions_len=32,
        leave_k_out=2,
    )
    metrics = pipeline.calc_ndcg_at_k(pipeline.leave_k_out)
    assert metrics["cover_ratio"] > 0.1


def test_cover_ratio():
    pipeline = PipelineML1M(
        "dataset/ml-1m",
        min_item_freq=1,
        min_user_interactions_len=5,
        max_user_interactions_len=32,
        leave_k_out=2,
    )
    metrics = pipeline.calc_ndcg_at_k(pipeline.leave_k_out)
    assert metrics["cover_ratio"] > 0.4
