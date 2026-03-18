# Continual contrastive reinforcement learning (CCRL)

Towards a stronger, environment-aware agent for commercial aero-engine fault diagnosis through long-term optimization under highly imbalanced scenarios.

English | [中文说明](./README.zh-CN.md)

`ccrl` is a Python package implementing the method described in the following paper:

> Wu, H., Zhong, S., Zhao, M., Fu, X., Zhang, Y., and Fu, S. Continual contrastive reinforcement learning: Towards stronger agent for environment-aware fault diagnosis of aero-engines through long-term optimization under highly imbalance scenarios. *Advanced Engineering Informatics*, 65, 103297 (2025). https://doi.org/10.1016/j.aei.2025.103297

## Paper and Project Links

- Paper: *Continual contrastive reinforcement learning: Towards stronger agent for environment-aware fault diagnosis of aero-engines through long-term optimization under highly imbalance scenarios*
- DOI: [https://doi.org/10.1016/j.aei.2025.103297](https://doi.org/10.1016/j.aei.2025.103297)
- Repository: [https://github.com/haozewu/ccrl](https://github.com/haozewu/ccrl)
- PyPI: [https://pypi.org/project/ccrl/](https://pypi.org/project/ccrl/)

The package is published on PyPI as `ccrl`. This repository contains its source code and documentation.

The CCRL pipeline implemented here includes:

- LSTM autoencoder pretraining
- contrastive representation learning
- D3QN-based type identification
- imbalanced reward design
- repeated imbalanced cross-validation

## Installation

```bash
pip install ccrl
```

For local development:

```bash
pip install .
```

## Quick Start

```python
from ccrl import CCRLConfig, run_ccrl_diag

config = CCRLConfig()
config.data.pretrain_label = "normal"

result = run_ccrl_diag(
    data_path="data/fault_dataset.pkl",
    config=config,
    repeats=1,
    seed=2024,
    log_dir="logs",
)

print(result.mean_f1, result.std_f1)
```

## CLI

```bash
ccrl --data data/fault_dataset.pkl --repeats 1 --seed 2024 --log-dir logs
```

## Input Data Format

The input file must be a pickle containing a `dict[str, samples]`.

- each key is a class label, such as `normal`, `fault_a`, or `bearing_outer`
- each value must be convertible to a 3D array with shape `(num_samples, seq_len, feature_dim)`
- all classes must share the same `seq_len` and `feature_dim`

Minimal example:

```python
{
    "normal": [
        [[0.1, 1.2], [0.2, 1.1], [0.3, 1.0]],
        [[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]],
    ],
    "fault_a": [
        [[1.2, 0.1], [1.1, 0.2], [1.0, 0.3]],
        [[0.9, 0.0], [0.8, 0.1], [0.7, 0.2]],
    ],
    "fault_b": [
        [[0.5, 2.0], [0.6, 1.9], [0.7, 1.8]],
        [[0.4, 2.1], [0.5, 2.0], [0.6, 1.9]],
    ],
}
```

This example has:

- `num_classes = 3`
- `seq_len = 3`
- `feature_dim = 2`

Save it as pickle:

```python
import pickle

data = {
    "normal": [
        [[0.1, 1.2], [0.2, 1.1], [0.3, 1.0]],
        [[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]],
    ],
    "fault_a": [
        [[1.2, 0.1], [1.1, 0.2], [1.0, 0.3]],
        [[0.9, 0.0], [0.8, 0.1], [0.7, 0.2]],
    ],
    "fault_b": [
        [[0.5, 2.0], [0.6, 1.9], [0.7, 1.8]],
        [[0.4, 2.1], [0.5, 2.0], [0.6, 1.9]],
    ],
}

with open("data/fault_dataset.pkl", "wb") as f:
    pickle.dump(data, f)
```

## Configuration Notes

```python
from ccrl import CCRLConfig

config = CCRLConfig()
config.data.class_order = ["normal", "fault_a", "fault_b"]
config.data.pretrain_label = "normal"
config.data.test_samples_per_class = 2
```

- `class_order` controls label encoding order and reporting order
- `pretrain_label` chooses which class is used for AE pretraining
- `test_samples_per_class` controls how many samples per class are drawn into the test split

## Notes

This repository is the source distribution of the `ccrl` package, intended for research, reproduction, and further development. The implementation here is an extracted and generalized offline package distilled from a larger formal system, rather than a full production deployment of that system. In particular, repository-level training scripts and APIs focus on the core CCRL modeling pipeline and evaluation workflow, not the complete business integration, online data flow, or continual update infrastructure described in the paper.

Unless explicitly stated otherwise, this repository should not be treated as the authors' complete production system. Because the code has been extracted, refactored, and generalized for reuse, it may contain engineering adaptations, simplifications, or implementation mistakes. If anything in this repository conflicts with the paper, the original paper should be treated as the authoritative reference.

## License

This project is released under the Apache License 2.0. See [LICENSE](./LICENSE).
