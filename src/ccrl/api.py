from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score

from .config import CCRLConfig
from .cv import ImbalanceCrossValidation
from .data import DatasetBundle, load_ccrl_pickle, normalization_for_lstm
from .game import FaultDiagnosisGame
from .models import DQN, DuelingLstmModel, LSTMAutoencoder, ProjectionHead, train_autoencoder, train_simclr
from .tracking import create_logger, create_writer


@dataclass(slots=True)
class FoldResult:
    fold_index: int
    mean_f1: float
    precision: list[float]
    recall: list[float]
    f1: list[float]
    kappa: float
    confusion_matrix: list[list[float]]


@dataclass(slots=True)
class RunResult:
    folds: list[FoldResult]
    mean_f1: float
    std_f1: float


def set_random_seed(random_seed: int):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)


def epsilon_increase(step, start=0.3, end=0.0001, max_value=10000):
    if step >= max_value:
        return end
    return start + (end - start) * (step / max_value)


def compute_imbalance_rate(train_index):
    type_values = [len(group) for group in train_index]
    denominator = np.sqrt(np.sum([1 / value**2 for value in type_values]))
    return [round((1 / value) / denominator, 4) for value in type_values]


def _run_process(
    dataset,
    model: DQN,
    usage: str,
    fold_index: int,
    episode: int,
    config: CCRLConfig,
    writer,
    imbalance_rate,
    label_names: list[str],
):
    sum_loss = 0.0
    sum_reward = 0.0
    learn_step = 0
    max_q_value = float("-inf")
    y_true = []
    y_pred = []
    epsilon = config.dqn.epsilon if config.dqn.epsilon is not None else config.dqn.initial_epsilon
    game = FaultDiagnosisGame(dataset, config, imbalance_rate)
    state = game.reset()
    while not game.finished:
        action = model.choose_action(state, usage, epsilon)
        if action["type"] == "net":
            max_q_value = max(max_q_value, action["value"].max(1)[0].item())
        next_state, reward, action_tensor, _, msg = game.step(action, usage)
        y_true.append(int(msg["fault_type"].item()))
        y_pred.append(int(action_tensor.item()))
        sum_reward += reward
        if usage == "train":
            learn_step += 1
            model.store_data(state, action_tensor, next_state, reward)
            if len(model.data_pool) > config.dqn.learn_when_capacity:
                loss = model.learn()
                sum_loss += loss
                global_step = episode * len(dataset) + learn_step
                writer.add_scalar(f"fold_{fold_index}/loss_step", loss, global_step)
                writer.add_scalar(f"fold_{fold_index}/epsilon_step", epsilon, global_step)
                if global_step % config.dqn.target_update_interval == 0:
                    model.target_net.load_state_dict(model.policy_net.state_dict())
                epsilon = epsilon_increase(global_step, start=config.dqn.initial_epsilon)
        state = next_state

    y_true_named = [label_names[label] for label in y_true]
    y_pred_named = [label_names[label] for label in y_pred]
    cm = confusion_matrix(y_true_named, y_pred_named, labels=label_names)
    precision = precision_score(y_true_named, y_pred_named, labels=label_names, average=None, zero_division=0)
    recall = recall_score(y_true_named, y_pred_named, labels=label_names, average=None, zero_division=0)
    f1 = f1_score(y_true_named, y_pred_named, labels=label_names, average=None, zero_division=0)
    kappa = cohen_kappa_score(y_true_named, y_pred_named)
    return {
        "confusion_matrix": cm,
        "sum_loss": sum_loss if sum_loss != 0 else np.inf,
        "sum_reward": sum_reward,
        "max_q_value": max_q_value if max_q_value != float("-inf") else 0.0,
        "y_true": y_true,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
    }


def run_single_fold(
    dataset_bundle: DatasetBundle,
    train_index,
    test_index,
    fold_index: int,
    config: CCRLConfig,
    writer,
    logger,
):
    data_arr = dataset_bundle.data_arr
    label_arr = dataset_bundle.label_arr
    label_names = dataset_bundle.label_names
    imbalance_rate = compute_imbalance_rate(train_index)
    trainset, testset = normalization_for_lstm(train_index, test_index, data_arr, label_arr, config)
    logger.info("fold %s train size %s test size %s imbalance %s", fold_index, len(trainset), len(testset), imbalance_rate)

    train_features, train_labels = trainset.tensors
    pretrain_label = config.data.pretrain_label
    pretrain_index = None
    if pretrain_label is not None and pretrain_label in label_names:
        pretrain_index = label_names.index(pretrain_label)
    else:
        for idx, label_name in enumerate(label_names):
            if label_name.lower() == "normal":
                pretrain_index = idx
                break
    if pretrain_index is None:
        class_sizes = [len(indexes) for indexes in train_index]
        pretrain_index = int(np.argmax(class_sizes))

    pretrain_samples = train_features[train_labels == pretrain_index]
    if len(pretrain_samples) == 0:
        pretrain_samples = train_features

    ae_model = LSTMAutoencoder(config.runtime.feature_dim, config.dqn.hidden_size, config.dqn.layers).to(config.runtime.device)
    train_autoencoder(ae_model, pretrain_samples, config, writer, logger)

    projection_input_dim = config.runtime.seq_len * config.dqn.hidden_size
    projection_hidden_dim = config.simclr.projection_hidden_dim or max(projection_input_dim // 2, config.runtime.num_classes)
    projection_output_dim = config.simclr.projection_output_dim or config.runtime.num_classes
    projection_head = ProjectionHead(projection_input_dim, projection_hidden_dim, projection_output_dim).to(config.runtime.device)
    train_simclr(ae_model.encoder, projection_head, trainset, config, writer, logger)

    network = DuelingLstmModel(
        ae_model.encoder,
        projection_head,
        config.dqn.hidden_size,
        config.dqn.layers,
        config.runtime.num_classes,
    )
    dqn_model = DQN(network, config)

    min_loss = np.inf
    no_min_loss_keep = 0
    last_episode = 0
    for episode in range(config.dqn.episodes):
        last_episode = episode
        dqn_model.policy_net.train()
        train_result = _run_process(
            trainset, dqn_model, "train", fold_index, episode, config, writer, imbalance_rate, label_names
        )
        writer.add_scalar(f"fold_{fold_index}/train_loss_episode", train_result["sum_loss"], episode)
        writer.add_scalar(f"fold_{fold_index}/train_reward_episode", train_result["sum_reward"], episode)
        writer.add_scalar(f"fold_{fold_index}/max_q_episode", train_result["max_q_value"], episode)
        if episode % 10 == 0:
            logger.info(
                "fold %s episode %s loss %.4f reward %.4f max_q %.4f",
                fold_index,
                episode,
                train_result["sum_loss"],
                train_result["sum_reward"],
                train_result["max_q_value"],
            )
        if episode > 0:
            if train_result["sum_loss"] < min_loss:
                min_loss = train_result["sum_loss"]
                no_min_loss_keep = 0
            else:
                no_min_loss_keep += 1
                if no_min_loss_keep > config.dqn.early_stop:
                    logger.info("fold %s early stop at episode %s", fold_index, episode)
                    break

    dqn_model.policy_net.eval()
    with torch.no_grad():
        test_result = _run_process(
            testset, dqn_model, "test", fold_index, last_episode, config, writer, imbalance_rate, label_names
        )

    return FoldResult(
        fold_index=fold_index,
        mean_f1=float(np.mean(test_result["f1"])),
        precision=test_result["precision"].tolist(),
        recall=test_result["recall"].tolist(),
        f1=test_result["f1"].tolist(),
        kappa=float(test_result["kappa"]),
        confusion_matrix=test_result["confusion_matrix"].tolist(),
    )


def run_ccrl_diag(
    data_path: str | Path,
    config: CCRLConfig | None = None,
    repeats: int = 1,
    seed: int = 2024,
    log_dir: str | Path | None = None,
) -> RunResult:
    config = config or CCRLConfig()
    set_random_seed(seed)
    logger = create_logger("ccrl", log_dir)
    writer = create_writer(log_dir)
    dataset_bundle = load_ccrl_pickle(data_path, class_order=config.data.class_order)
    config.runtime.label_names = dataset_bundle.label_names
    config.runtime.num_classes = len(dataset_bundle.label_names)
    config.runtime.seq_len = dataset_bundle.seq_len
    config.runtime.feature_dim = dataset_bundle.feature_dim
    min_class_size = min(len(samples) for samples in dataset_bundle.data_arr)
    if min_class_size < 2:
        raise ValueError("Each class must contain at least 2 samples to create separate train/test splits.")
    test_samples_per_class = min(config.data.test_samples_per_class, min_class_size - 1)
    rkf = ImbalanceCrossValidation(n_splits=test_samples_per_class, n_repeats=repeats, random_state=seed)
    folds = []
    try:
        for fold_index, train_index, test_index in rkf.split(dataset_bundle.data_arr):
            fold_result = run_single_fold(
                dataset_bundle=dataset_bundle,
                train_index=train_index,
                test_index=test_index,
                fold_index=fold_index,
                config=config,
                writer=writer,
                logger=logger,
            )
            folds.append(fold_result)
    finally:
        writer.close()

    mean_f1 = float(np.mean([fold.mean_f1 for fold in folds])) if folds else float("nan")
    std_f1 = float(np.std([fold.mean_f1 for fold in folds])) if folds else float("nan")
    logger.info("completed %s folds mean_f1 %.4f std %.4f", len(folds), mean_f1, std_f1)
    return RunResult(folds=folds, mean_f1=mean_f1, std_f1=std_f1)
