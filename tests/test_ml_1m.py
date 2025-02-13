import sys

sys.path.append("src")

from fastEASE import Dataset, PipelineEASE
import pytest
from collections.abc import Iterable


class DatasetML1M(Dataset):
    def __init__(self, path_to_dataset: str):
        super().__init__(self.load_interactions(path_to_dataset))

    @staticmethod
    def load_interactions(path_to_dataset) -> Iterable[tuple[int, int]]:
        path_to_interactions = path_to_dataset + "/" + "ratings.dat"
        with open(path_to_interactions, "r") as file:
            for line in file:
                yield tuple(map(int, line.strip("\n").split("::")[:2]))


@pytest.fixture
def dataset_ml_1m():
    return DatasetML1M("dataset/ml-1m")


def test_items_vocab(dataset_ml_1m):
    assert len(dataset_ml_1m.items_vocab) > 1000


def test_users_vocab(dataset_ml_1m):
    assert len(dataset_ml_1m.users_vocab) > 500


def test_interactions_matrix(dataset_ml_1m):
    assert dataset_ml_1m.interactions_matrix.shape == (
        len(dataset_ml_1m.users_vocab),
        len(dataset_ml_1m.items_vocab),
    )


def test_ndcg():
    pipeline = PipelineEASE(
        user_item_it=DatasetML1M.load_interactions("dataset/ml-1m"),
        min_item_freq=1,
        min_user_interactions_len=5,
        max_user_interactions_len=32,
        calc_ndcg_at_k=True,
        k=5,
        predict_next_n=False,
    )
    assert pipeline.ndcg > 0.2
