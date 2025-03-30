""" Module for the model. """

import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

TIMESTAMP = datetime.now()


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
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


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

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.float)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()

        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

    def save(self):
        """
        Saves the current state of the model and optimizer to a .pth file in the 'models' folder.
        Uses the current time to name the file.
        """
        timestamp = TIMESTAMP.strftime("%Y-%m-%d_%H-%M-%S")

        folder = 'models'
        file_name = 'model_' + timestamp + '.pth'
        file_path = os.path.join(folder, file_name)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, file_path)
        print(f'Model saved to {file_path}')

    def load(self, path):
        """
        Loads the model and optimizer from a file.

        :param path: The path to the .pth file.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Model and optimizer loaded from {path}')
