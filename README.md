# Policy Gradient Implementations

Implementing algorithms from https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

## Prerequisites

Python3

## Running an experiment

From the root directory:

```sh
python3 policy_gradients <experiment>
```

Currently available experiments:

-   actor_critic
-   ddpg
-   reinforce
-   sac
-   td3

## Troubleshooting

I had trouble installing some packages (e.g. `pybullet`) directly using Pipenv. This worked for me:

1. Start a Pipenv shell: `pipenv shell`
2. Install `pybullet` using `pip`: `pip install pybullet`
3. Add `pybullet` to `Pipfile` manually
4. Generate lockfile: `pipenv lock`
