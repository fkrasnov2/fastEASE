import sys
sys.path.append("src")

from fastEASE import Dataset
import  pytest
from collections.abc import Iterable 

class DatasetML1M (Dataset):
    def __init__(self, path_to_dataset: str):
        super().__init__(self._load_interactions(path_to_dataset)) 

    def _load_interactions(self, path_to_dataset) -> Iterable[tuple[int,int]]:
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
    assert dataset_ml_1m.interactions_matrix.shape == (len(dataset_ml_1m.users_vocab), len(dataset_ml_1m.items_vocab))

