import torch
import random
import numpy as np
from collections import deque

from game import Game
from model import Linear_QNet, QTrainer
# from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class Agent:

    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0    # randomness
        self.gamma = 0.9    # discount rate (must be smaller than 1)
        self.memory = deque(maxlen=MAX_MEMORY)  # if full -> popleft
        # self.model = Linear_QNet(11, 256, 3)
        # self.model.load()
        # self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)

    def get_state(self, game: Game) -> np.ndarray:
        next_checkpoint = game.get_next_checkpoint()
        rays = game.get_rays()

        state = [
            # Rays
            *rays,

            # Move direction
            game.car.velocity,
            game.car.angle,

            # TODO include distance from next checkpoint ?
            # TODO include the threshold here ?
            # Next checkpoint location
            game.car.x_position < next_checkpoint[0],
            game.car.x_position > next_checkpoint[0],
            game.car.y_position < next_checkpoint[1],
            game.car.y_position > next_checkpoint[1],
        ]

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        ...

    def train_long_memory(self):
        ...

    def train_short_memory(self, state, action, reward, next_state, done):
        ...

    def get_action(self, state):

        return random.randint(0, 8)


def train():
    agent = Agent()
    game = Game()

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

            # if score > record:
            #     record = score
            #     agent.model.save()

            # print(f"Game: {agent.number_of_games}, Score: {score}, Record: {record}")

            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score / agent.number_of_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
