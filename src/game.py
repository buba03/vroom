import pygame
# import random
# import numpy as np
from collections import namedtuple

from car import Car
from utils.enums import *

pygame.init()
font = pygame.font.SysFont('arial', 25)


SPEED = 30

Position = namedtuple('Position', 'x, y')


class CarGameAI:
    def __init__(self, screen_width=900, screen_height=600):
        # Screen
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Car
        self.car = Car(CarID.MCLAREN.value)

        # init display
        self.display = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Vroom v1.0.0')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        pass

    def play_step(self, action):

        reward = 0
        game_over = None
        score = None

        if action == 1:
            self.car.accelerate()
            self.car.rotate(10)
        elif action == 2:
            self.car.accelerate()
        elif action == 3:
            self.car.accelerate()
            self.car.rotate(-10)
        elif action == 4:
            self.car.rotate(-10)
        elif action == 5:
            self.car.brake()
            self.car.rotate(10)
        elif action == 6:
            self.car.brake()
        elif action == 7:
            self.car.brake()
            self.car.rotate(-10)
        elif action == 8:
            self.car.rotate(10)
        else:
            pass

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, score

    def _update_ui(self):
        # Background
        self.display.fill(Color.BLACK.value)

        # Items
        self.car.draw(self.display)

        # Text
        text = font.render("Velocity: " + str(self.car.velocity), True, Color.WHITE.value)
        self.display.blit(text, [0, 0])
        text = font.render("Angle: " + str(self.car.angle), True, Color.WHITE.value)
        self.display.blit(text, [0, 30])
        text = font.render(f"Position: x: {str(int(self.car.x_position))} y: {str(int(self.car.y_position))}", True, Color.WHITE.value)
        self.display.blit(text, [0, 60])

        # Update
        pygame.display.flip()


if __name__ == '__main__':
    game = CarGameAI()

    while True:
        action = 0

        keys = pygame.key.get_pressed()

        accelerate = keys[pygame.K_w] or keys[pygame.K_UP]
        brake = keys[pygame.K_s] or keys[pygame.K_DOWN]
        turn_left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        turn_right = keys[pygame.K_d] or keys[pygame.K_RIGHT]

        if accelerate and brake:
            action = 0
        elif accelerate and turn_left and not turn_right:
            action = 1
        elif accelerate and ((not turn_left and not turn_right) or (turn_left and turn_right)):
            action = 2
        elif accelerate and turn_right and not turn_left:
            action = 3
        elif turn_right and (not accelerate and not brake):
            action = 4
        elif brake and turn_right and not turn_left:
            action = 5
        elif brake and (not turn_left and not turn_right):
            action = 6
        elif brake and turn_left and not turn_right:
            action = 7
        elif turn_left and (not accelerate or not brake):
            action = 8

        game.play_step(action)
