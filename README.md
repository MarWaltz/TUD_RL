# RL Dresden Algorithm Suite

This suite implements several model-free off-policy deep reinforcement learning algorithms for discrete and continuous action spaces in PyTorch.

## Algorithms

| Name             | Single-/Multi-Agent |  Action Space |                                                                                                  Source |
| ---------------- | :--------------------:|  :----------: | ------------------------------------------------------------------------------------------------------: |
| DQN              |     Single    |   Discrete   |                                        [Mnih et. al. 2015](https://www.nature.com/articles/nature14236) |
| Double DQN       |     Single    |   Discrete   |                              [van Hasselt et. al. 2016](https://dl.acm.org/doi/10.5555/3016100.3016191) |
| Bootstrapped DQN |     Single    |   Discrete   |                                                 [Osband et. al. 2016](https://arxiv.org/abs/1602.04621) |
| Ensemble DQN     |     Single    |   Discrete   |                                 [Anschel et. al 2017](http://proceedings.mlr.press/v70/anschel17a.html) |
| MaxMin DQN       |     Single    |   Discrete   |                                                    [Lan et. al. 2020](https://arxiv.org/abs/2002.06487) |
| SCDQN            |     Single    |   Discrete   |                                [Zhu et. al. 2021](https://www.aaai.org/AAAI21Papers/AAAI-3820.ZhuR.pdf) |
| ACCDDQN          |     Single    |   Discrete   |                                                  [Jiang et. al. 2021](https://arxiv.org/abs/2105.00704) |
| KE-BootDQN       |     Single    |   Discrete   |                                                  [Waltz, Okhrin 2022](https://arxiv.org/abs/2201.08078) |
|                  |
| DDPG             |     Single    |  Continuous  |                                              [Lillicrap et. al. 2015](https://arxiv.org/abs/1509.02971) |
| LSTM-DDPG        |     Single    |  Continuous  |                              [Meng et. al. 2021](https://ieeexplore.ieee.org/abstract/document/9636140) |
| TD3              |     Single    |  Continuous  |                             [Fujimoto et. al. 2018](https://proceedings.mlr.press/v80/fujimoto18a.html) |
| LSTM-TD3         |     Single    |  Continuous  |                              [Meng et. al. 2021](https://ieeexplore.ieee.org/abstract/document/9636140) |
| SAC              |     Single    |  Continuous  |                                               [Haarnoja et. al. 2019](https://arxiv.org/abs/1812.05905) |
| LSTM-SAC         |     Single    |  Continuous  | Own Implementation following [Meng et. al. 2021](https://ieeexplore.ieee.org/abstract/document/9636140) |
| TQC              |     Single    |  Continuous  |                           [Kuznetsov et. al. 2020](http://proceedings.mlr.press/v119/kuznetsov20a.html) |
|                  |
| MADDPG           |     Multi    |  Continuous  |                                       [Lowe et. al. 2017](https://proceedings.neurips.cc/paper/2017/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html)
| MATD3            |     Multi    |  Continuous  |                                       [Ackermann et. al. 2019](https://arxiv.org/abs/1910.01465)
|                  |
| DiscMADDPG           |     Multi    |  Discrete  |                                       Gumbel-Softmax discretization of MADDPG
| DiscMATD3            |     Multi    |  Discrete  |                                       Gumbel-Softmax discretization of MATD3

## Prerequisites

To use basic functions of this package you need to have at least installed

- [OpenAI Gym](https://github.com/openai/gym)
- [PyTorch](https://github.com/pytorch/pytorch)

In order to use the package to its full capabilites, it is recommended to install the following dependencies:

- [MinAtar](https://github.com/kenjyoung/MinAtar)
- [PyGame Learning Environment](https://pygame-learning-environment.readthedocs.io/en/latest/user/games.html)
- [Gym-Games](https://github.com/qlan3/gym-games)

## Installation

The package is set up to be used as an editable install, which makes prototyping very easy and does not require you to rebuild the package after every change.

Install it using pip:

```bash
$ git clone https://github.com/MarWaltz/TUD_RL.git
$ cd TUD_RL/
$ pip install -e .
```

> Note that a normal package install via pip is not supported at the moment and will lead to import errors.

## Usage

### Configuration files

In order to train in an environment using this package, you must specify a training configuration `.yaml` file and place it in one of the two folders in `/tud_rl/configs` depending on the type of action space (discrete, continuous).

You also find a variety of different example configuration files in this folder.

For an increased flexibility, please make yourself familiar with the parameters each algorithm offers.

### Training

The recommended way to train or visualize your environment is to use the `tud_rl` package as a module using the `python -m` flag.

To run the package, you have to supply the following flags to the module:

##### -m [--mode=]

Training mode can be either `train` or `visualize`. If you want to visualize your environment, you must ensure that training weights are supplied in the config file:

For discrete training the config entry looks like:

```yaml
---
dqn_weights: /path/to/weights.pth
```

For continuous training you must supply both actor and critic weights:

```yaml
---
actor_weights: /path/to/actor_weights.pth
critic_weights: /path/to/critic_weights.pth
```

##### -c [--config_file=]

Name of your configuration file placed in either `/tud_rl/configs/discrete_actions` or `/tud_rl/configs/continuous_actions`.

##### -a [--agent_name=]

Name of the agent you want to use for training or visualization. The specified agent must be a present in your configuration file.

#### Example:

```bash
$ python -m tud_rl -m train -c myconfig.yaml -a DDQN
```

## Gym environment integration

This package provides an interface to specify your own custom training environment based on the OpenAI framework. Once this is done, no further adjustment is needed and you can start training as described in the section above.

### Structure

In order to integrate your own environment you have to create a new file in `/tud_rl/envs/_envs`. There you need to specify a class for your environment that implements at least three methods as seen in the following blueprint:

#### Empty custom env [minimal example]

```python
# This file is named Dummy.py
import gym
class MyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        """Your code"""

    def reset():
        """reset your env"""
        pass

    def step(action):
        """Perform step in environment"""
        pass

    def render(): # optional
        """Render your env to an output"""
        pass
```

See this [blog article](https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952) for a detailed explanation on how to set up your own gym environment

### Integration of custom environment into TUD_RL

Once your environment is specified, you need to register it with gym in order to add it to the list of callable environments. The registration is done in the `/tud_rl/__init__.py` file by selecting the name your environment will be called with, and the entry point for gym to know where your custom environment is located (loc is the fixed base location while the rest is the class name of your environment):

```python
register(
    id="MyEnv-v0",
    entry_point= loc + "MyEnv",
)
```

You can now select your environment in your configuration file under the `env` category.

Example (incomplete):

```yaml
---
env:
  name: MyEnv-v0
  max_episode_steps: 100
  state_type: feature
  wrappers: []
  wrapper_kwargs: {}
  env_kwargs: {}
  info: ""
agent:
  DQN: {}
```

## Citation

If you use this code in one of your projects or papers, please cite it as follows.

```bibtex
@misc{TUDRL,
  author = {Waltz, Martin and Paulig, Niklas},
  title = {RL Dresden Algorithm Suite},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/MarWaltz/TUD_RL}}
}
```
