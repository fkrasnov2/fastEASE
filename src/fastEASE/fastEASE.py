"""The module makes it possible to use EASE as part of a CUDA application."""

import time
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from scipy.sparse import csr_matrix, identity, lil_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ndcg_score
from tqdm import tqdm

try:
    import cupy as cp

    cp.cuda.is_available()
except Exception:
    print("CUDA is not available")
    cp = np


class Dataset:
    def __init__(
        self,
        user_item_it: Iterable[tuple[str, str]],
        min_item_freq: int = 3,
        min_user_interactions_len: int = 3,
        max_user_interactions_len: int = 32,
    ) -> None:
        """Create a dataset from list of interactions (user, item)"""
        self._user_item_it = user_item_it
        self.min_item_freq = min_item_freq
        self.min_user_interactions_len = min_user_interactions_len
        self.max_user_interactions_len = max_user_interactions_len
        self._interactions_matrix, self._users_vocab, self._items_vocab = (
            self._convert_user_item_list_to_interactions_matrix()
        )

    @property
    def items_vocab(self) -> np.array:
        return self._items_vocab

    @property
    def users_vocab(self) -> list:
        return self._users_vocab

    @property
    def interactions_matrix(self) -> csr_matrix:
        return self._interactions_matrix

    def _convert_user_item_list_to_interactions_matrix(self) -> None:
        """
        Convert list of tuples (user_id,item_id) to interactions matrix with users x items.
        Filter out low freq items aka "long tail".
        Filter short interactions sequences.
        Save interaction_matrix to class instance property.

        Returns
        -------
        None
        """
        user_history = defaultdict(list)
        vocab = defaultdict(int)
        if not isinstance(self._user_item_it, Iterable):
            raise ValueError("user_item_it is not iterable")

        for user_id, item_id in tqdm(self._user_item_it):
            user_history[user_id].append(item_id)
            vocab[item_id] += 1

        vocab = dict(filter(lambda item: item[1] >= self.min_item_freq, vocab.items()))

        print(f"ALL : {len(user_history)=}")
        user_history = dict(
            filter(
                lambda item: len(item[1]) >= self.min_user_interactions_len
                and len(item[1]) < self.max_user_interactions_len,
                map(
                    lambda item: (
                        item[0],
                        list(filter(lambda item_id: item_id in vocab, item[1])),
                    ),
                    user_history.items(),
                ),
            )
        )

        print(f"FILTERED : {len(user_history)=}")
        self.cv = CountVectorizer(
            lowercase=False,
            tokenizer=lambda items: items,
            token_pattern=None,
            dtype=np.int32,
        )
        return (
            self.cv.fit_transform(list(user_history.values())),
            list(user_history.keys()),
            self.cv.get_feature_names_out(),
        )

    def random_split(
        self, interactions_matrix: csr_matrix, k: int = 2
    ) -> tuple[csr_matrix, csr_matrix]:
        """randomly choose k items (columns) as test, and erase them from train matrix"""
        _, item_num = interactions_matrix.shape
        train_items = np.random.randint(0, item_num, size=k)
        train = interactions_matrix.tolil()
        train[:, train_items] = 0

        test = interactions_matrix[:, train_items]

        return train.tocsr(), test

    def leave_k_last_split(
        self, interactions_matrix: csr_matrix, k: int = 2
    ) -> tuple[csr_matrix, csr_matrix]:
        """For each user, leave the last k interacted items as test data, removing them from the training matrix."""
        # Convert to LIL format for efficient row manipulations
        train = interactions_matrix.tolil()
        num_users, num_items = train.shape
        test = lil_matrix((num_users, num_items), dtype=interactions_matrix.dtype)

        for user in tqdm(range(num_users)):
            cols = train.rows[user]
            data = train.data[user]
            num_interactions = len(cols)
            if num_interactions == 0:
                continue  # No interactions to split

            # Determine the number of items to move to test (up to k)
            # So one can use another rule instead of 0
            # e.g:
            # if num_interactions < k:
            #     split = num_interactions // 2
            # split = max(1, split)
            split = max(0, num_interactions - k)

            # Split the indices and data
            train_cols = cols[:split]
            train_data = data[:split]
            test_cols = cols[split:]
            test_data = data[split:]

            # Update the train and test matrices
            train.rows[user] = train_cols
            train.data[user] = train_data
            test.rows[user] = test_cols
            test.data[user] = test_data

        # Convert back to CSR format before returning
        return train.tocsr(), test.tocsr()

    def metrics(self, test : csr_matrix, prediction :  np.array, k : int) -> dict:
        prediction_csr = self.cv.transform(prediction)
        score = ~(test != prediction_csr).toarray() * 1
        y_score = score.copy()
        score.sort(axis=1)
        y_true = np.fliplr(score)
        ndcg = ndcg_score(y_true, y_score, k=k).item()

        # calc diversity ratio
        distinct_inference_size = np.unique(prediction.ravel()).shape[0]
        distinct_test_size = np.unique(test.indices).shape[0]
        diversity_ratio = 1.0 * distinct_inference_size / distinct_test_size
        self._metrics = {
            f"nDCG@{k}": ndcg,
            "distinct_inference_size": distinct_inference_size,
            "diversity_ratio": diversity_ratio,
        }

        return self._metrics

        
class Model:
    def __init__(self, interactions_matrix: csr_matrix, regularization: float) -> None:
        self._regularization = regularization
        self._interactions_matrix = interactions_matrix
        self._weights_matrix = self._fit()

    @property
    def weights_matrix(self) -> cp.array:
        return self._weights_matrix

    def _fit(self) -> cp.array:
        """Fit interaction matrix to weights_matrix"""
        colocations = self._interactions_matrix.T @ self._interactions_matrix
        colocations += self._regularization * identity(colocations.shape[0])
        colocations_inv = cp.linalg.inv(cp.array(colocations.toarray()))
        start_time = time.perf_counter()
        weights_matrix = colocations_inv / (-np.diag(colocations_inv))
        print(f"inv finished: {time.perf_counter() - start_time}")
        cp.fill_diagonal(weights_matrix, 0.0)
        return weights_matrix

    def predict_next_n(
        self,
        interactions_matrix: csr_matrix,
        prediction_batch_size: int = 1000,
        next_n: int = 12,
    ) -> np.array:
        """Predict next n items for each user in interactions matrix.
        Split on batches to save memory.
        """
        if self._weights_matrix is None:
            raise ValueError("Model not fit")

        start_time = time.perf_counter()
        print("predict started")
        inferenced_item_ids_list = []
        users_number, _ = interactions_matrix.shape

        for batch_index in tqdm(range(0, users_number, prediction_batch_size)):
            interaction_batch = cp.array(
                interactions_matrix[
                    batch_index : batch_index + prediction_batch_size
                ].toarray()
            )
            predicted_batch = cp.argsort(interaction_batch.dot(self.weights_matrix))[
                :, -next_n:
            ]
            del interaction_batch
            inferenced_item_ids_list.append(
                predicted_batch.get()
                if not isinstance(predicted_batch, np.ndarray)
                else predicted_batch
            )
        inferenced_item_ids = np.vstack(inferenced_item_ids_list)
        print(f"predict finished: {time.perf_counter() - start_time}")

        return inferenced_item_ids


class PipelineEASE(Dataset):
    def __init__(
        self,
        user_item_it: Iterable[tuple[str, str]],
        min_item_freq: int = 3,
        min_user_interactions_len: int = 3,
        max_user_interactions_len: int = 32,
    ) -> None:
        """Init and pipeline execution"""
        super().__init__(
            user_item_it,
            min_item_freq,
            min_user_interactions_len,
            max_user_interactions_len,
        )
        
        print(f"{self.interactions_matrix.shape=}")

    def calc_ndcg_at_k(self, k : int = 3, regularization : int = 100, prediction_batch_size : int = 1000, ) -> dict:
            train, test = self.leave_k_last_split(self.interactions_matrix, k=k)
            model = Model(train, regularization=regularization)
            prediction = model.predict_next_n(
                interactions_matrix=train,
                prediction_batch_size=prediction_batch_size,
                next_n=k,
            )
            return self.metrics(test, prediction, k)

    def predict_next_n(self,  next_n : int = 3, return_items : bool = False, prediction_batch_size : int = 1000, regularization : int = 100 ) -> np.array:
            model = Model(self.interactions_matrix, regularization=regularization)
            prediction = model.predict_next_n(
                interactions_matrix=self.interactions_matrix,
                prediction_batch_size=prediction_batch_size,
                next_n=next_n,
            )
            if return_items:
                prediction = self.items_vocab[prediction]
            users = np.array(self.users_vocab).reshape((-1, 1))
            return np.hstack((users, prediction))

