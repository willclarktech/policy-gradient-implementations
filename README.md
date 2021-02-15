# Policy Gradient Implementations

Implementing reinforcement learning algorithms based on policy gradients.

## Prerequisites

-   Python3.6

## Installation

Using poetry:

```sh
poetry install
```

Using pip:

```sh
pip install -r requirements.txt
```

Using pip to install development dependencies too:

```sh
pip install -r requirements.dev.txt
```

On Google Colab to avoid conflicts with preinstalled packages:

```sh
pip install -r requirements.colab.txt
```

## Running experiments

### CLI

An executable is provided in `./bin`. From the root directory run:

```sh
./bin/policy_gradients <algorithm>
```

To see the full list of options, including available algorithms:

```sh
./bin/policy_gradients --help
```

Several pre-trained models are provided in `./models`. For example, to view a pre-trained SAC agent operate in the `InvertedPendulumBulletEnv-v0` environment you can run:

```sh
./bin/policy_gradients sac -n 1 --env InvertedPendulumBulletEnv-v0 --eval --render --load_dir ./models
```

### Programmatic API

Use the exposed `run` function with an options dictionary. This will be combined with a set of default hyperparameters for the relevant algorithm. For example:

```py
import policy_gradients

policy_gradients.run({
    "algorithm": "sac",
    "env_name": "LunarLanderContinuous-v2",
    "n_episodes": 250,
    "log_period": 10,
    "save_dir": "./models",
    "seed": 123456,
})
```

Refer to `parser.py` for the full list of available options as well as the `hyperparameters.py` file for the relevant algorithm to see which hyperparameters apply.

### Notebooks

A set of notebooks is provided in `./notebooks` which demonstrates how to train each algorithm for an appropriate environment using the programmatic API. Each notebook provides a link to open the notebook in Google Colab. To run locally start a Jupyter notebook server and open the relevant notebook in the browser window which should open automatically:

```sh
jupyter notebook
```

## Development

The following scripts assume the requirements have been installed. If using poetry, they assume `poetry shell` has already been run or else they should be prefixed with `poetry run`.

### Lint

```sh
pylint ./policy_gradients
```

### Typecheck

```sh
mypy
```

### Format

```sh
black ./policy_gradients
```

### Generating requirements files

```sh
./scripts/generate_requirements.sh
```

## Troubleshooting

### Poetry

I had trouble installing `gym` with Poetry because of its `Pillow` dependency and something to do with `zlib`. Setting `PKG_CONFIG_PATH="/usr/local/opt/zlib/lib/pkgconfig"` fixed this problem.
