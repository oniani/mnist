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
$ python train.py  # or ./train.py
```

Bonus: [WandB](https://wandb.ai/) takes care of generating the loss plot and saving the training
configuration.

## Model Evaluation

```console
$ python eval.py  # or ./eval.py
```

## License

[MIT License](LICENSE)
