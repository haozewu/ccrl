# ccrl

[English](./README.md) | 中文说明

`ccrl` 是对以下论文方法的 Python 包实现：

Haoze Wu, Shisheng Zhong, Minghang Zhao, Xuyun Fu, Yongjian Zhang, Song Fu, "Continual contrastive reinforcement learning: Towards stronger agent for environment-aware fault diagnosis of aero-engines through long-term optimization under highly imbalance scenarios", *Advanced Engineering Informatics*, DOI: `10.1016/j.aei.2025.103297`。

- 仓库地址：`https://github.com/haozewu/ccrl`
- PyPI：`https://pypi.org/project/ccrl/0.1.0/`
- 论文 DOI：`10.1016/j.aei.2025.103297`

`ccrl` 已发布到 PyPI：`https://pypi.org/project/ccrl/0.1.0/`，这个仓库包含它的源码和说明文档。

当前实现覆盖的 CCRL 核心流程包括：

- LSTM 自编码器预训练
- 对比学习表征训练
- 基于 D3QN 的故障类型识别
- 面向不均衡样本的奖励设计
- 重复不均衡交叉验证

## 安装

```bash
pip install ccrl
```

本地开发安装：

```bash
pip install .
```

## 快速使用

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

## 命令行

```bash
ccrl --data data/fault_dataset.pkl --repeats 1 --seed 2024 --log-dir logs
```

## 输入数据格式

输入文件必须是一个 pickle，内容为 `dict[str, samples]`。

- 每个 `key` 是类别名，例如 `normal`、`fault_a`、`bearing_outer`
- 每个 `value` 必须能转换成三维数组，形状为 `(num_samples, seq_len, feature_dim)`
- 所有类别必须共享相同的 `seq_len` 和 `feature_dim`

最小示例：

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

这个示例对应：

- `num_classes = 3`
- `seq_len = 3`
- `feature_dim = 2`

保存为 pickle：

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

## 配置说明

```python
from ccrl import CCRLConfig

config = CCRLConfig()
config.data.class_order = ["normal", "fault_a", "fault_b"]
config.data.pretrain_label = "normal"
config.data.test_samples_per_class = 2
```

- `class_order` 控制标签编码顺序和结果展示顺序
- `pretrain_label` 指定哪一类样本用于 AE 预训练
- `test_samples_per_class` 控制每次划分时每类抽取多少测试样本

## 说明

这个仓库是 `ccrl` 包的源码仓库，面向研究复现、方法验证和二次开发。这里提供的是从更大正式系统中剥离、整理并通用化后的离线实现，重点覆盖 CCRL 的核心建模流程和评估流程，并不等同于论文场景中的完整生产系统。也就是说，论文中提到的业务集成、在线数据流、经验库联动、持续更新基础设施等内容，并没有被完整打包进当前仓库。

除非另有明确说明，这个仓库不应被理解为论文作者正式系统的完整对外发布版本。由于这套代码经过了整理、抽取、重构和通用化处理，过程中可能引入工程化改写、简化处理或实现偏差。如果本仓库内容与论文表述不一致，应以论文原文为准。

## 协议

本项目采用 Apache License 2.0 开源协议。见 [LICENSE](./LICENSE)。
