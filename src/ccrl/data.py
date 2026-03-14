from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn import preprocessing
from torch.utils.data import TensorDataset

from .config import CCRLConfig


@dataclass(slots=True)
class DatasetBundle:
    data_arr: list[np.ndarray]
    label_arr: list[np.ndarray]
    label_names: list[str]
    seq_len: int
    feature_dim: int


def _validate_samples(group_name: str, values) -> np.ndarray:
    samples = np.asarray(values)
    if samples.ndim != 3:
        raise ValueError(f"Samples for '{group_name}' must be a 3D array-like object of shape (n, seq_len, feature_dim).")
    return samples


def load_ccrl_pickle(data_path: str | Path, class_order: list[str] | None = None) -> DatasetBundle:
    with open(data_path, "rb") as file:
        data = pickle.load(file)

    if not isinstance(data, dict) or not data:
        raise ValueError("Input pickle must contain a non-empty dict of label name -> samples.")

    if class_order is not None:
        label_names = class_order
        missing = [label for label in label_names if label not in data]
        if missing:
            raise ValueError(f"Missing labels in input data: {missing}")
        data_arr = [_validate_samples(label, data[label]) for label in label_names]
    else:
        label_names = []
        data_arr = []
        ignored_keys = []
        for label, values in data.items():
            samples = np.asarray(values)
            if samples.ndim == 3:
                label_names.append(label)
                data_arr.append(samples)
            else:
                ignored_keys.append(label)
        if not data_arr:
            raise ValueError("No valid sample groups found. Expected at least one 3D array-like value with shape (n, seq_len, feature_dim).")

    sample_shape = data_arr[0].shape[1:]
    for label, samples in zip(label_names, data_arr):
        if samples.shape[1:] != sample_shape:
            raise ValueError(
                f"All classes must share the same sample shape. '{label}' has {samples.shape[1:]}, expected {sample_shape}."
            )

    label_arr = [
        np.full(len(samples), label_idx, dtype=np.int32)
        for label_idx, samples in enumerate(data_arr)
    ]
    return DatasetBundle(
        data_arr=data_arr,
        label_arr=label_arr,
        label_names=label_names,
        seq_len=sample_shape[0],
        feature_dim=sample_shape[1],
    )


def build_labels(train_index, test_index, label_arr):
    train_label = []
    test_label = []
    for i in range(len(label_arr)):
        train_label.append(label_arr[i][train_index[i]])
        test_label.append(label_arr[i][test_index[i]])
    return (
        np.hstack(tuple(train_label)),
        np.array(test_label).reshape(-1),
    )


def scale_splits(train_index, test_index, data_arr):
    train_parts = []
    test_parts = []
    for i in range(len(data_arr)):
        train_parts.append(data_arr[i][train_index[i]])
        test_parts.append(data_arr[i][test_index[i]])

    train_data = np.concatenate(train_parts, axis=0)
    test_data = np.concatenate(test_parts, axis=0)

    scaler = preprocessing.MinMaxScaler()
    feature_dim = train_data.shape[-1]
    train_flat = np.reshape(train_data, (-1, feature_dim))
    test_flat = np.reshape(test_data, (-1, feature_dim))
    scaler.fit(train_flat)

    train_scaled = scaler.transform(train_flat).reshape(train_data.shape)
    test_scaled = scaler.transform(test_flat).reshape(test_data.shape)
    return train_scaled, test_scaled


def normalization_for_lstm(train_index, test_index, data_arr, label_arr, config: CCRLConfig):
    train_scaled, test_scaled = scale_splits(train_index, test_index, data_arr)
    train_label, test_label = build_labels(train_index, test_index, label_arr)
    device = config.runtime.device
    return (
        TensorDataset(torch.from_numpy(train_scaled).float().to(device), torch.from_numpy(train_label).long().to(device)),
        TensorDataset(torch.from_numpy(test_scaled).float().to(device), torch.from_numpy(test_label).long().to(device)),
    )
