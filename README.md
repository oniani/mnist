# The Reproducible MNIST

Reproducible [MNIST](https://en.wikipedia.org/wiki/MNIST_database) experiments in
[PyTorch](https://pytorch.org/).

## Setup

```console
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ wandb login API_KEY
```

## Model Training

```console
$ CUBLAS_WORKSPACE_CONFIG=:16:8 python train.py  # or CUBLAS_WORKSPACE_CONFIG=:16:8 ./train.py
```

You should set the environment variable `CUBLAS_WORKSPACE_CONFIG` according to
[CUDA documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility).

Bonus: [WandB](https://wandb.ai/) takes care of generating the loss plot and saving the training
configuration.

## Model Evaluation

```console
$ python eval.py  # or ./eval.py
```

## License

[MIT License](LICENSE)
