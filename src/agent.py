""" Module for the agent. When ran as main, the agent will start training. """

import os
import random
from collections import deque

import numpy as np
import torch

from game import Game, GameAction
from model import LinearQNet, QTrainer
from utils.config_manager import ConfigManager
from utils.statistics import training_plot, debug_plot

# TODO put this in a yaml
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
GAMMA = 0.995
EPSILON = 1.0
# EPSILON_DECAY = 0.9995  # ~4600 episodes
EPSILON_DECAY = 0.9977  # ~1000 episodes
# EPSILON_DECAY = 0.99    # ~230 episodes
MIN_EPSILON = 0.1

STATE_ATTRIBUTE_COUNT = 10
HIDDEN_LAYER = 256


def init_model(model_name):
    """
    Initializes a new or existing LinearQNet and QTrainer and returns it.

    :param model_name: The name of the model inside the 'models' folder.
    :return: The new or existing LinearQNet and QTrainer as a tuple.
    """
    network = LinearQNet(STATE_ATTRIBUTE_COUNT, HIDDEN_LAYER, GameAction().action_count)
    trainer = QTrainer(network, lr=LEARNING_RATE, gamma=GAMMA)

    folder = 'models'
    path = os.path.join(folder, model_name)
    if os.path.exists(path) and model_name != '':
        trainer.load(path)
        print(f'Loaded model from {path}')
    elif model_name == '':
        print('Starting from scratch...')
    else:
        raise FileNotFoundError(f'No saved model found at {path}')

    return network, trainer


class Agent:
    """ Class for an agent to train the model by playing the game. """

    def __init__(self, model_name: str = ""):
        """
        Initializes the agent with a new or existing model.

        :param model_name: The name of the model inside the 'models' folder.
        Default is an empty string, which will create a new LinearQNet.
        """

        self.episode_count = 0
        self.epsilon = EPSILON
        # TODO fine-tune gamma
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)  # if full -> automatic popleft
        self.model, self.trainer = init_model(model_name)

    @staticmethod
    def get_state(game: Game) -> np.ndarray:
        """
        Returns the state of the game.
        It includes:
            the length of the cast rays,
            the car's normalized velocity and angle,
            the car's positional relation to the next checkpoint.

        :param game: The game to get the state of.
        :return: The state of the game as a numpy array.
        """
        # Get normalized rays (0 to 1)
        rays = game.get_rays(normalize=True)

        # Normalize velocity (0 to 1)
        velocity = game.car.velocity / game.car.max_speed

        # Separate angle into sin and cos values
        angle_sin = np.sin(np.radians(game.car.angle))
        angle_cos = np.cos(np.radians(game.car.angle))

        # Normalize distances (0 to 1)
        x_distance, y_distance = game.get_distance_from_next_checkpoint()
        x_distance = x_distance / game.display.get_size()[0]
        y_distance = y_distance / game.display.get_size()[1]

        state = [
            *rays,
            velocity,
            angle_sin,
            angle_cos,
            x_distance,
            y_distance
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
        # Empty action
        game_action = GameAction()

        # Epsilon-greedy (exploration / exploitation)
        if random.uniform(0, 1) < self.epsilon:
            # Exploration
            move = random.randint(0, game_action.action_count - 1)
            game_action.action = move
        else:
            # Exploitation
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            game_action.action = move

        return game_action.action

    def apply_epsilon_decay(self):
        """ Applies epsilon decay to reduce exploration over time. """
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)


# When ran as main, the agent will start the training process.
if __name__ == '__main__':
    car_arg = ConfigManager().get_argument('car')
    track_arg = ConfigManager().get_argument('track')
    model_arg = ConfigManager().get_argument('model')

    agent = Agent(model_arg)
    game = Game(car_arg, track_arg)

    MAX_STEPS = 15_000
    record = 0
    counter = 0

    # For plotting
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    plot_epsilon = []
    plot_steps = []

    while True:
        counter += 1
        # Current state
        state_old = agent.get_state(game)

        # Choose action
        final_move = agent.get_action(state_old)

        # Apply action
        reward, done, score = game.play_step(final_move)
        # Get the new state
        state_new = agent.get_state(game)

        # Single training step
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # Store experience for 'experience replay'
        agent.remember(state_old, final_move, reward, state_new, done)

        if done or counter > MAX_STEPS:
            game.reset()
            agent.episode_count += 1

            # Epsilon decay
            agent.apply_epsilon_decay()
            # Experience replay
            agent.train_long_memory()

            # Save model, if improved
            if score > record:
                record = score
                agent.trainer.save()

            print(f"Game: {agent.episode_count}, Score: {score}, Record: {record}")

            # Plotting
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.episode_count
            plot_mean_scores.append(mean_score)
            training_plot(plot_scores, plot_mean_scores)

            plot_epsilon.append(agent.epsilon)
            plot_steps.append(counter / MAX_STEPS)
            debug_plot(plot_epsilon, plot_steps)

            counter = 0
