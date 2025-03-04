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
    def __init__(self, **kwargs):
        """
        Create a dataset from list of interactions (user, item)
        """
        self._user_item_it: Iterable[tuple[str, str]] = kwargs.get("user_item_it", [])
        self.min_item_freq: int = kwargs.get("min_item_freq", 3)
        self.min_user_interactions_len: int = kwargs.get("min_user_interactions_len", 3)
        self.max_user_interactions_len: int = kwargs.get(
            "max_user_interactions_len", 32
        )
        self.leave_k_out = kwargs.get("leave_k_out", 0)
        # processes
        (
            self._cv,
            self._interactions_matrix,
            self._leave_k_out_matrix,
            self._users_vocab,
            self._items_vocab,
        ) = self._convert_user_history_to_interactions_matrix(
            self._make_user_history(), self.leave_k_out
        )

    @property
    def items_vocab(self) -> np.array:
        return self._items_vocab

    @property
    def users_vocab(self) -> np.array:
        return self._users_vocab

    @property
    def interactions_matrix(self) -> csr_matrix:
        return self._interactions_matrix

    @property
    def leave_k_out_matrix(self) -> np.array:
        return self._leave_k_out_matrix

    def _make_user_history(
        self,
    ) -> dict[str, list]:
        """
        Convert list of tuples (user,item) to user history dict  {user: items}.
        Filter out low freq items aka "long tail".
        Filter short interactions sequences.

        Returns
        -------
        dict {user:items}
        """
        user_history = defaultdict(list)
        vocab = defaultdict(int)

        if not isinstance(self._user_item_it, Iterable):
            raise ValueError("user_item_it is not iterable")

        for user, item in tqdm(self._user_item_it):
            items = user_history.get(user, [])

            if len(items) == 0:
                user_history[user] = [item]
                vocab[item] += 1

            elif items[-1] != item:  # protect from repeets
                user_history[user].append(item)
                vocab[item] += 1

        # min frequency vocab filter
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
        return user_history

    def _convert_user_history_to_interactions_matrix(
        self, user_history: dict, leave_k_out: int
    ) -> tuple[CountVectorizer, csr_matrix, np.array, np.array, np.array]:
        """
        Convert dict of user histories to csr_matrix (interaction matrix).
        If leave_k_out > 0 create leave_k_out_matrix with the same shape containing last k items from each user history.

        Returns
        -------

        CountVectorizer object
        interactions_matrix -- sparse interactions matrix,
        leave_k_out_matrix -- np.array with shape (users,leave_k_out)
        user2id, item2id -- np.arrays with users and items from axis of interactions_matrix

        """
        _vocab = set()
        _users = []
        train_user_history = []
        test_user_history = []

        for user, items in user_history.items():
            if len(set(items)) < leave_k_out * 2:
                continue

            _users.append(user)
            if leave_k_out > 0:
                test_user_history.append(items[-leave_k_out:])
                train_user_history.append(items[:-leave_k_out])
            else:
                train_user_history.append(items)
            _vocab.update(items)

        assert len(_users) == len(set(_users))
        user2id = {user: idx for idx, user in enumerate(_users)}

        ## Tokenization
        item2id = {item: idx for idx, item in enumerate(_vocab)}
        train_items = []
        test_items = []

        if leave_k_out > 0:
            for train, test in zip(train_user_history, test_user_history):
                train_items.append(list(map(lambda item: item2id[item], train)))
                test_items.append(list(map(lambda item: item2id[item], test)))
        else:
            for train in train_user_history:
                train_items.append(list(map(lambda item: item2id[item], train)))

        print(f"{len(train_items)=} {len(test_items)=}")

        cv = CountVectorizer(
            lowercase=False,
            tokenizer=lambda items: items,
            token_pattern=None,
            dtype=np.int32,
            vocabulary=list(range(len(item2id))),
        )
        cv.fit(train_items)
        return (
            cv,
            cv.transform(train_items),
            np.array(test_items),
            np.array(list(user2id.keys())),
            np.array(list(item2id.keys())),
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

    def sparse_leave_k_last_split(
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


class Metrics(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def ndcg(self, test: np.array, prediction: np.array, k: int) -> dict[str, float]:
        k = k or test.shape[1]
        score = (test[:, :k] == prediction[:, :k]) * 1
        y_score = score.copy()
        score.sort(axis=1)
        y_true = np.fliplr(score)
        return {f"nDCG@{k}": ndcg_score(y_true, y_score, k=k).item()}

    def cover_ratio(self, test: np.array, prediction: np.array) -> dict[str, float]:
        """
        calc cover ratio
        """
        distinct_inference_size = np.unique(prediction.ravel()).shape[0]
        distinct_test_size = np.unique(test.ravel()).shape[0]
        cover_ratio = 1.0 * distinct_inference_size / distinct_test_size
        metrics = {
            "distinct_inference_size": distinct_inference_size,
            "cover_ratio": cover_ratio,
        }

        return metrics


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


class PipelineEASE(Metrics):
    def __init__(self, **kwargs):
        """Init and pipeline execution"""
        super().__init__(**kwargs)
        print(f"{self.interactions_matrix.shape=}")

    def calc_ndcg_at_k(
        self,
        k: int = 3,
        regularization: int = 100,
        prediction_batch_size: int = 1000,
    ) -> dict:
        model = Model(self.interactions_matrix, regularization=regularization)
        prediction = model.predict_next_n(
            interactions_matrix=self.interactions_matrix,
            prediction_batch_size=prediction_batch_size,
            next_n=k,
        )

        ndcg = self.ndcg(self.leave_k_out_matrix, prediction, k)
        cover_ratio = self.cover_ratio(self.leave_k_out_matrix, prediction)
        return {**ndcg, **cover_ratio}

    def predict_next_n(
        self,
        next_n: int = 5,
        return_items: bool = False,
        prediction_batch_size: int = 1000,
        regularization: int = 100,
    ) -> np.array:
        model = Model(self.interactions_matrix, regularization=regularization)
        prediction = model.predict_next_n(
            interactions_matrix=self.interactions_matrix,
            prediction_batch_size=prediction_batch_size,
            next_n=next_n,
        )
        if return_items:
            prediction = self.items_vocab[prediction]
        users = self.users_vocab.reshape((-1, 1))
        return np.hstack((users, prediction))
