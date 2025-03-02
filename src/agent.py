""" Module for the agent. When ran as main, the agent will start training. """

import random
from collections import deque

import numpy as np
import torch

from game import Game, GameAction
from model import Linear_QNet, QTrainer
from utils.config_manager import ConfigManager

# from helper import plot

# TODO put this in a yaml
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class Agent:
    """ Class for an agent to train the model by playing the game. """

    def __init__(self):
        """ Initializes the agent. """

        self.number_of_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate (must be smaller than 1)
        self.memory = deque(maxlen=MAX_MEMORY)  # if full -> popleft
        self.model = Linear_QNet(9, 256, GameAction().action_count)
        # self.model.load()
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)

    @staticmethod
    def get_state(game: Game) -> np.ndarray:
        """
        Returns the state of the game.
        It includes:
            the length of the cast rays,
            the car's velocity and angle,
            the car's positional relation to the next checkpoint.

        :param game: The game to get the state of.
        :return: The state of the game as a numpy array.
        """
        distances_from_next_checkpoint = game.get_distance_from_next_checkpoint()
        rays = game.get_rays()

        state = [
            *rays,
            game.car.velocity,
            game.car.angle,
            *distances_from_next_checkpoint
        ]

        return np.array(state, dtype=float)

    def remember(self, state: np.ndarray, action: list[int], reward: float, next_state: np.ndarray, done: bool):
        """
        Stores an experience tuple in the agent's memory for replay during training.

        Appends a given game experience to the memory buffer.
        If the memory buffer reaches its maximum size, the oldest experiences are removed.

        :param state: The state of the game before the action.
        :param action: The action taken.
        :param reward: The reward after the action.
        :param next_state: The state of the game after the action.
        :param done: Whether the game is over or not.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        Train the agent using a batch of experiences stored in memory.

        Randomly samples a batch from the agent's memory to perform a training step.
        If the memory size is smaller than the batch size, the entire memory is used.
        """
        if len(self.memory) > BATCH_SIZE:
            # Random sample
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            # Full memory
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state: np.ndarray, action: list[int], reward: float, next_state: np.ndarray,
                           done: bool):
        """
        Performs a single training step using the given experience.

        :param state: The state of the game before the action.
        :param action: The action taken.
        :param reward: The reward after the action.
        :param next_state: The state of the game after the action.
        :param done: Whether the game is over or not.
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: np.ndarray) -> list[int]:
        """
        Returns an action based on the given state.

        :param state: The state of the game.
        :return: The action chosen by the model.
        """
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.number_of_games
        # Empty action
        game_action = GameAction()
        # Random
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 8)
            game_action.action = move
        # Model
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # executes the model.forward function
            move = torch.argmax(prediction).item()
            game_action.action = move

        return game_action.action


# When ran as main, the agent will start the training process.
if __name__ == '__main__':
    # TODO extend docs
    """
    Starts the training process. Instantiates a fresh Agent and Game.

    """
    # TODO add changeable game parameters
    agent = Agent()
    record = 0

    car_arg = ConfigManager().get_argument('car')
    track_arg = ConfigManager().get_argument('track')
    game = Game(car_arg, track_arg)

    # plot_scores = []
    # plot_mean_scores = []
    # total_score = 0

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train the long memory (experience replay), plot result
            game.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f"Game: {agent.number_of_games}, Score: {score}, Record: {record}")

            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score / agent.number_of_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)
