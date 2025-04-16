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
EPSILON_DECAY = 0.9995  # ~4600 episodes
# EPSILON_DECAY = 0.9977  # ~1000 episodes
# EPSILON_DECAY = 0.99    # ~230 episodes
MIN_EPSILON = 0.1

STATE_ATTRIBUTE_COUNT = 10
HIDDEN_LAYER_1 = 256
HIDDEN_LAYER_2 = 256


def init_model(model_name):
    """
    Initializes a new or existing LinearQNet and QTrainer and returns it.

    :param model_name: The name of the model inside the 'models' folder.
    :return: The new or existing LinearQNet and QTrainer as a tuple.
    """
    network = LinearQNet(STATE_ATTRIBUTE_COUNT, HIDDEN_LAYER_1, HIDDEN_LAYER_2, GameAction().action_count)
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

    def __init__(self, model_name: str = '', eval: bool = False):
        """
        Initializes the agent with a new or existing model.

        :param model_name: The name of the model inside the 'models' folder.
        Default is an empty string, which will create a new LinearQNet.
        :param eval: Whether the agent should only evaluate the model or start a training.
        If True, model_name is required. Default is False.
        """
        self.eval = eval
        if eval and model_name == '':
            raise RuntimeError('Eval mode requires a model name.')

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

        # Normalized angle difference from next checkpoint
        angle_difference = game.get_angle_difference_from_checkpoint(game.get_next_checkpoint(), normalize=True)

        # Normalize velocity (0 to 1)
        velocity = game.car.velocity / game.car.max_speed

        # Normalize distances (0 to 1)
        x_distance, y_distance = game.get_distance_from_next_checkpoint()
        x_distance = x_distance / game.display.get_size()[0]
        y_distance = y_distance / game.display.get_size()[1]

        state = [
            # TODO more state attributes
            # can see checkpoint? (maybe front 3 rays?)
            *rays,
            angle_difference,
            velocity,
            # x_distance,
            # y_distance
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
    eval_mode = ConfigManager.get_argument('eval')

    agent = Agent(model_arg, bool(eval_mode))
    game = Game(car_arg, track_arg)

    if not agent.eval:
        record = 0  # all-time best score
        MAX_STEPS = 10_000  # max steps per episode
        steps = 0  # for counting steps per episode
        n = 100  # last n games' scores will be saved
        recent_avg_record = 0  # last n games' best avg. score

        # For plotting
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        current_total_reward = 0
        plot_rewards = []
        plot_mean_rewards = []
        total_rewards = 0
        plot_epsilon = []
        plot_steps = []

        print('Starting training...')
        while True:
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

            # Update values
            steps += 1
            current_total_reward += reward

            if done or steps > MAX_STEPS:
                game.reset()
                agent.episode_count += 1

                # Epsilon decay
                agent.apply_epsilon_decay()
                # Experience replay
                agent.train_long_memory()

                # Save model, if improved
                if score > record:
                    record = score
                    agent.trainer.save(f'{record}pts')

                # Plotting
                plot_scores.append(score)
                total_score += score
                plot_mean_scores.append(total_score / agent.episode_count)

                plot_rewards.append(current_total_reward / 100)
                total_rewards += current_total_reward
                plot_mean_rewards.append(total_rewards / 100 / agent.episode_count)

                training_plot(plot_scores, plot_mean_scores, plot_rewards, plot_mean_rewards)

                plot_epsilon.append(agent.epsilon)
                plot_steps.append(steps / MAX_STEPS)
                debug_plot(plot_epsilon, plot_steps)

                # Reset values
                steps = 0
                current_total_reward = 0

                last_n_game_avg_score = sum(plot_scores[-n:]) // n
                print(f'Game: {agent.episode_count}, Score: {score}, Record: {record}. '
                      f'Last {n} game avg. score: {last_n_game_avg_score}, and record: {recent_avg_record}')

                # Save model if avg. improved
                if last_n_game_avg_score > recent_avg_record:
                    recent_avg_record = last_n_game_avg_score
                    agent.trainer.save(f'{record}avg')

    else:
        print('Running in evaluation mode...')
        agent.epsilon = 0.0  # only exploitation
        episode_count = 10  # number of episode to evaluate
        MAX_STEPS = 10_000  # max steps per episode
        MAX_LAPS = 10  # max laps to record per episode
        steps = 0  # for counting steps per episode

        for i in range(episode_count):
            steps = 0
            game.reset()

            print(f'[Episode {i + 1}]', end=' ')

            while True:
                state = agent.get_state(game)
                action = agent.get_action(state)
                _, done, score = game.play_step(action)

                steps += 1

                if steps >= MAX_STEPS:
                    print(f'Reached max steps ({MAX_STEPS}). Score: {score}, Laps: {game.lap_count}')
                    break
                if done:
                    print(f'Game over. Score: {score}, Laps: {game.lap_count}')
                    break
                if game.lap_count >= MAX_LAPS:
                    print(f'Successfully completed {MAX_LAPS} laps. Score: {score}, Steps: {steps}')
                    break
