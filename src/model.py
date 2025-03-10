""" Module for the model. """

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim


class LinearQNet(nn.Module):
    """
    A simple fully connected neural network model for Q-learning.

    This model contains:
    - An input layer that accepts the state of the game.
    - One hidden layer with ReLU activation.
    - An output layer that produces the Q-values for each possible action.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initializes the LinearQNet.

        :param input_size: The size of the input layer (number of features in the state).
        :param hidden_size: The number of neurons in the hidden layer.
        :param output_size: The size of the output layer (number of possible actions).
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: The input tensor representing the state.
        :return: A tensor representing the Q-values for each possible action.
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self):
        """
        Saves the current state of the model to a file in the 'models' folder.
        Uses the current timestamp.
        """
        folder = 'models'
        file_name = 'model_' + str(time.time()) + '.pth'
        file_name = os.path.join(folder, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, path):
        """
        Loads the model state from a file.

        :param path: The path to the model.
        """
        self.load_state_dict(torch.load(path, weights_only=False))


class QTrainer:
    """ A class for training and optimizing the Q-learning model. """

    def __init__(self, model, lr, gamma):
        """
        Initializes the QTrainer.

        :param model: The Q-learning model to be trained.
        :param lr: The learning rate for the optimizer.
        :param gamma: The discount factor for future rewards.
        """
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state: np.ndarray, action: list[int], reward: float, next_state: np.ndarray, done: bool):
        """
        Performs a single training step using the provided training data.
        Parameters can be single values or lists.

        :param state: The state of the game before the action.
        :param action: The action taken.
        :param reward: The reward after the action.
        :param next_state: The state of the game after the action.
        :param done: Whether the game is over or not.
        """
        state = torch.tensor(np.array(state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.float)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)

        if len(state.shape) == 1:  # Handle single-sample input
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done,)

        # Predicted Q-values from the current state
        pred = self.model(state)

        # Update Q-values using the Bellman equation
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
