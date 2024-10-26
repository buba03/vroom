""" Module for game logic. When ran as main, the game will use the player controls. """

import pygame
# import random
# import numpy as np

from car import Car
from track import Track
from utils.enums import Direction, Color, CarID, TrackID

# Initialize pygame
pygame.init()
# Constants
FONT = pygame.font.SysFont('arial', 25)
FPS = 30


class Game:
    """ Class for game logic. """

    def __init__(self, screen_width: int = 900, screen_height: int = 600):
        """
        Sets up the game.

        :param screen_width:
        :param screen_height:
        """
        # Screen
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Car
        self.car = Car(CarID.FERRARI.value)
        # Track
        self.track = Track(TrackID.SIMPLE.value)

        # init display
        self.display = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Vroom v1.0.0')
        # Clock
        self.clock = pygame.time.Clock()
        # Set up elements
        self.reset()

    def reset(self):
        """ Sets up the default game, resets the game state. """
        x, y, angle = self.track.get_car_default_state()
        self.car.reset(x, y, angle)

    def play_step(self, action) -> tuple[float, bool, float]:
        """
        Play the next step of the game using the given action.

        :param action: The action to execute in the next step of the game.
        :return: reward (for the executed action), game_over (whether the continues), score (the current score)
        """
        # TODO: reward, game_over, score system
        reward = 0
        game_over = self._check_collision()
        score = 0

        if game_over:
            self.reset()

        if action == 1:
            self.car.accelerate()
            self.car.turn(Direction.LEFT)
        elif action == 2:
            self.car.accelerate()
        elif action == 3:
            self.car.accelerate()
            self.car.turn(Direction.RIGHT)
        elif action == 4:
            self.car.turn(Direction.RIGHT)
        elif action == 5:
            self.car.brake()
            self.car.turn(Direction.RIGHT)
        elif action == 6:
            self.car.brake()
        elif action == 7:
            self.car.brake()
            self.car.turn(Direction.LEFT)
        elif action == 8:
            self.car.turn(Direction.LEFT)
        else:
            pass

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # TODO: plots?

        # TODO: move before or after action ?
        self.car.move()
        self._update_display()
        self.clock.tick(FPS)
        return reward, game_over, score

    def _update_display(self):
        """ Update the display. """

        # Background
        self.display.fill(Color.GRASS.value)

        # Items
        self.track.draw(self.display)
        self.car.draw(self.display)
        # self.display.blit(pygame.mask.from_surface(pygame.transform.rotate(self.car.image, self.car.angle)).to_surface(), (self.car.x_position, self.car.y_position))

        # Text
        text = FONT.render("Velocity: " + str(self.car.velocity), True, Color.WHITE.value)
        self.display.blit(text, [0, 0])
        text = FONT.render("Angle: " + str(self.car.angle), True, Color.WHITE.value)
        self.display.blit(text, [0, 30])
        text = FONT.render(f"Position: x: {str(int(self.car.x_position))} y: {str(int(self.car.y_position))}", True, Color.WHITE.value)
        self.display.blit(text, [0, 60])

        # Update
        pygame.display.flip()

    def _check_collision(self) -> bool:
        # FIXME
        # TODO docs
        # TODO returns true only when completely left the track, should return true when touched the side of the track ?
        car_mask = pygame.mask.from_surface(self.car.image)
        track_mask = pygame.mask.from_surface(self.track.image)

        return not bool(car_mask.overlap(track_mask, (0-self.car.x_position, 0-self.car.y_position)))


# When ran as main, the game will use player inputs.
if __name__ == '__main__':

    game = Game()

    # Game loop
    while True:
        # Default action (do nothing)
        player_action = 0

        # Currently pressed keys
        keys = pygame.key.get_pressed()

        # WASD and arrow key presses to actions
        accelerate = keys[pygame.K_w] or keys[pygame.K_UP]
        brake = keys[pygame.K_s] or keys[pygame.K_DOWN]
        turn_left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        turn_right = keys[pygame.K_d] or keys[pygame.K_RIGHT]

        # Calculate action
        if accelerate and brake:
            player_action = 0
        elif accelerate and turn_left and not turn_right:
            player_action = 1
        elif accelerate and ((not turn_left and not turn_right) or (turn_left and turn_right)):
            player_action = 2
        elif accelerate and turn_right and not turn_left:
            player_action = 3
        elif turn_right and (not accelerate and not brake):
            player_action = 4
        elif brake and turn_right and not turn_left:
            player_action = 5
        elif brake and (not turn_left and not turn_right):
            player_action = 6
        elif brake and turn_left and not turn_right:
            player_action = 7
        elif turn_left and (not accelerate or not brake):
            player_action = 8

        # Execute action, go to next state
        game.play_step(player_action)
