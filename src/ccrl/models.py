from __future__ import annotations

import copy
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import CCRLConfig

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden_state, _) = self.encoder(x)
        decoder_input = hidden_state[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(decoder_input)
        return decoded


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, dim=1)


class DuelingLstmModel(nn.Module):
    def __init__(self, encoder_lstm, projection_head, hidden_size, num_layers, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.lstm = nn.LSTM(encoder_lstm.hidden_size, hidden_size, num_layers, batch_first=True)
        for network in [encoder_lstm, projection_head]:
            for param in network.parameters():
                param.requires_grad = False
        encoder_lstm.eval()
        self.ae_lstm = encoder_lstm
        self.value_stream = nn.Linear(hidden_size, 1)
        self.advantage_stream = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x, _ = self.ae_lstm(x)
        _, (hidden_state, _) = self.lstm(x)
        x = hidden_state[-1, :, :]
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class DQN(nn.Module):
    def __init__(self, network: nn.Module, config: CCRLConfig):
        super().__init__()
        self.config = config
        self.policy_net = copy.deepcopy(network).to(config.runtime.device)
        self.target_net = copy.deepcopy(network).to(config.runtime.device)
        self.target_net.eval()
        self.data_pool = deque([], maxlen=config.dqn.memory_capacity)
        self.loss_func = nn.MSELoss().to(config.runtime.device)
        self.optimizer = optim.Adam(
            [
                {"params": self.policy_net.value_stream.parameters(), "lr": config.dqn.learning_rate},
                {"params": self.policy_net.advantage_stream.parameters(), "lr": config.dqn.learning_rate},
                {"params": self.policy_net.lstm.parameters(), "lr": config.dqn.learning_rate},
            ]
        )

    def choose_action(self, state, usage, epsilon):
        if usage == "train" and np.random.uniform() < epsilon:
            return {
                "type": "random",
                "value": torch.tensor(
                    [[random.randrange(self.policy_net.num_actions)]],
                    device=self.config.runtime.device,
                    dtype=torch.long,
                ),
            }
        with torch.no_grad():
            return {"type": "net", "value": self.policy_net(state)}

    def store_data(self, *args):
        self.data_pool.append(Transition(*args))

    def learn(self):
        transitions = random.sample(self.data_pool, self.config.dqn.mini_batch)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.config.runtime.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.as_tensor(batch.reward, dtype=torch.float32).to(self.config.runtime.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_actions = torch.zeros(self.config.dqn.mini_batch, device=self.config.runtime.device, dtype=torch.long)
        next_state_actions[non_final_mask] = self.policy_net(non_final_next_states).argmax(1).detach()
        next_state_values = torch.zeros(self.config.dqn.mini_batch, device=self.config.runtime.device)
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states)
            .gather(1, next_state_actions[non_final_mask].unsqueeze(1))
            .squeeze(1)
            .detach()
        )
        expected_state_action_values = (next_state_values * self.config.dqn.gamma) + reward_batch

        loss = self.loss_func(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


def train_autoencoder(model: LSTMAutoencoder, train_data: torch.Tensor, config: CCRLConfig, writer, logger):
    min_loss = np.inf
    keep_times = 0
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.dqn.autoencoder_lr)
    dataloader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=False)

    for epoch in range(config.dqn.autoencoder_epochs):
        model.train()
        running_loss = 0.0
        for batch_data in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        writer.add_scalar("loss/autoencoder", running_loss, epoch + 1)
        if epoch % 10 == 0:
            logger.info("AE epoch %s loss %.4f", epoch + 1, running_loss)
        if running_loss < min_loss:
            min_loss = running_loss
            keep_times = 0
        else:
            keep_times += 1
            if keep_times >= config.dqn.autoencoder_patience:
                logger.info("AE early stop at epoch %s", epoch + 1)
                break


def get_positive_negative_pairs(features, labels):
    positive_pairs = []
    negative_pairs = []
    batch_size = features.size(0)
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if labels[i] == labels[j]:
                positive_pairs.append((features[i], features[j]))
            else:
                negative_pairs.append((features[i], features[j]))
    return positive_pairs, negative_pairs


def contrastive_loss(positive_pairs, negative_pairs, config: CCRLConfig):
    temperature = config.simclr.temperature
    pos_weight = config.simclr.positive_weight
    neg_weight = config.simclr.negative_weight
    loss = 0.0
    negative_similarities = [
        torch.exp(neg_weight * F.cosine_similarity(neg1.unsqueeze(0), neg2.unsqueeze(0), dim=-1) / temperature)
        for neg1, neg2 in negative_pairs
    ]
    sum_negative_similarities = sum(negative_similarities)
    for pos1, pos2 in positive_pairs:
        positive_similarity = F.cosine_similarity(pos1.unsqueeze(0), pos2.unsqueeze(0), dim=-1) / temperature
        loss -= pos_weight * torch.log(
            torch.exp(positive_similarity) / (torch.exp(positive_similarity) + sum_negative_similarities)
        )
    return loss / max(len(positive_pairs), 1)


def train_simclr(encoder_lstm: nn.LSTM, projection_head: ProjectionHead, train_data, config: CCRLConfig, writer, logger):
    optimizer = optim.Adam(
        [
            {"params": encoder_lstm.parameters(), "lr": config.simclr.encoder_lr},
            {"params": projection_head.parameters(), "lr": config.simclr.projector_lr},
        ]
    )
    data_loader = DataLoader(train_data, batch_size=config.simclr.batch_size, shuffle=True)
    min_loss = np.inf
    keep_times = 0
    for epoch in range(config.simclr.epochs):
        running_loss = 0.0
        projection_head.train()
        for features, labels in data_loader:
            optimizer.zero_grad()
            encoded, _ = encoder_lstm(features)
            lstm_features = encoded.flatten(1, 2)
            projections = projection_head(lstm_features)
            positive_pairs, negative_pairs = get_positive_negative_pairs(projections, labels)
            if not positive_pairs or not negative_pairs:
                continue
            loss = contrastive_loss(positive_pairs, negative_pairs, config)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        writer.add_scalar("loss/simclr", running_loss, epoch + 1)
        if epoch % 10 == 0:
            logger.info("SimCLR epoch %s loss %.4f", epoch + 1, running_loss)
        if running_loss < min_loss:
            min_loss = running_loss
            keep_times = 0
        else:
            keep_times += 1
            if keep_times >= config.simclr.early_stop_patience:
                logger.info("SimCLR early stop at epoch %s", epoch + 1)
                break
