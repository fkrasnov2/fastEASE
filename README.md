# fast EASE for CUDA
[![Python version](https://img.shields.io/badge/Python-%3E=3.10-blue)](https://github.com/fkrasnov2/fastEASE)

##
**EASY** is a unique and decisive approach to recommendation for a limited number of users and items.
One challenge is that the matrix inversion process becomes computationally intensive, requiring significant processing time on the central processing unit (CPU).
This issue is addressed in the current project by leveraging CUDA, a powerful technology specifically designed for parallel processing. The key distinction is that this solution is intended not for research purposes but rather for deployment in production environments.

## Framework

![Framework](https://media.githubusercontent.com/media/fkrasnov2/fastEASE/main/fastEASEv2.png)

## Structure

- `.github`: GitHub Actions workflows.
- `src`: Library's source code.
- `tests`: Unit tests.
- `.editorconfig`: Editor settings for consistent coding style.
- `.gitignore`: Excludes files generated by Python and Poetry.
- `LICENSE`: License file.
- `Makefile`: Manage tasks like testing, linting, and formatting.
- `pyproject.toml`: PyPi's configuration file.

## Getting Started
bash:
```console
pip install fastEASE
mkdir dataset
wget --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-1m.zip -O dataset/ml-1m.zip
unzip dataset/ml-1m.zip -d dataset/
```
python:
```python
k = 5
pipeline = PipelineEASE(
    user_item_it=DatasetML1M.load_interactions(
        "dataset/ml-1m"
    ),
    min_item_freq=1,
    min_user_interactions_len=5,
    max_user_interactions_len=32,
    calc_ndcg_at_k=True,
    k=k,
    predict_next_n=False,
)
print(f"nDCG@{k} = {pipeline.ndcg:.4}")

```

### Prerequisites

- `Python` >= 3.10
- `GNU Make`

Tested on `Ubuntu 24.04 LTS` and `Debian 12`. But the template should work on other operating systems as well.

### Setting Things Up

1. **Clone the repository**:

    ```sh
    git clone https://github.com/fkrasnov2/fastEASE
    cd fastEASE
    ```

2. **Install dependencies**:
    ```sh
    make install
    ```

### Development Workflow Management
```sh
# Run the unit tests
make test
```

```sh
# Lint the code
make lint
```


