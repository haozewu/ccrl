from __future__ import annotations

from torch import nn
from torch.utils.data import DataLoader

from .config import CCRLConfig


def give_reward(predicted, actual, imbalance_rate):
    if predicted == actual:
        return imbalance_rate[int(actual)]
    return -1 * imbalance_rate[int(actual)]


class FaultDiagnosisGame:
    def __init__(self, dataset, config: CCRLConfig, imbalance_rate):
        self.config = config
        self.imbalance_rate = imbalance_rate
        self.loss_func = nn.CrossEntropyLoss().to(config.runtime.device)
        self.train_loader = DataLoader(
            dataset,
            shuffle=True,
            num_workers=config.runtime.num_workers,
            pin_memory=config.runtime.pin_memory,
        )

    def reset(self):
        self.data_iter = iter(self.train_loader)
        self.finished = False
        single_sample = next(self.data_iter)
        self.answer = single_sample[1]
        return single_sample[0]

    def step(self, action, usage):
        self.finished = len(self.train_loader) == self.data_iter._num_yielded
        if action["type"] == "net":
            loss = self.loss_func(action["value"], self.answer)
            action_tensor = action["value"].max(1)[1].view(1, 1)
        else:
            loss = 0
            action_tensor = action["value"]
        reward = give_reward(action_tensor, self.answer, self.imbalance_rate)
        if reward > 0 and self.finished:
            reward += self.config.dqn.reward_plus
        msg = {"fault_type": self.answer, "is_correct": reward > 0}
        if not self.finished:
            single_sample = next(self.data_iter)
            self.answer = single_sample[1]
            return single_sample[0], reward, action_tensor, loss, msg
        return None, reward, action_tensor, loss, msg
