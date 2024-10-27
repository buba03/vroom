""" Module for game logic. When ran as main, the game will use the player controls. """

import os
import pygame
# import random
# import numpy as np

from car import Car
from track import Track
from utils.enums import Direction, Color, CarID, TrackID
from utils.yaml_manager import YamlManager

# Initialize pygame
pygame.init()
# Constants
FONT = pygame.font.SysFont('arial', 25)


class Game:
    """ Class for game logic. """

    def __init__(self, car: str = CarID.FERRARI.value, track: str = TrackID.SIMPLE.value):
        """
        Sets up the game.

        :param car: The name of the car inside the cars.yaml file.
        :param track: The name of the track inside the tracks.yaml file.
        """
        # yaml import
        fps, display_attributes = YamlManager(os.path.join('resources', 'game.yaml')).get_game_attributes()
        # Display
        self.display_width = display_attributes['width']
        self.display_height = display_attributes['height']
        # FPS
        self.fps = fps

        # Car
        self.car = Car(car)
        # Track
        self.track = Track(track)

        # init display
        self.display = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption('Vroom v1.0.0')
        # Clock
        self.clock = pygame.time.Clock()
        # Set up elements
        size = YamlManager(os.path.join('resources', 'tracks.yaml')).get_car_size_from_track(self.track.id)
        self.car.resize(size)
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
        game_over = self._car_offtrack()
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
        self.clock.tick(self.fps)
        # print(self.clock.get_fps())
        return reward, game_over, score

    def _update_display(self):
        """ Update the display. """

        # Background
        self.display.fill(Color.GRASS.value)

        # Items
        self.track.draw(self.display)
        self.car.draw(self.display)

        # Text
        text = FONT.render("Velocity: " + str(self.car.velocity), True, Color.WHITE.value)
        self.display.blit(text, [0, 0])
        text = FONT.render("Angle: " + str(self.car.angle), True, Color.WHITE.value)
        self.display.blit(text, [0, 30])
        text = FONT.render(f"Position: x: {str(int(self.car.x_position))} y: {str(int(self.car.y_position))}", True, Color.WHITE.value)
        self.display.blit(text, [0, 60])
        text = FONT.render(f"FPS: {str(self.clock.get_fps())}", True, Color.WHITE.value)
        self.display.blit(text, [0, 90])

        # Update
        pygame.display.flip()

    def _car_offtrack(self) -> bool:
        """
        Checks whether the track and the car has collided.

        :return: True if the car has completely left the track, False otherwise.
        """
        # TODO maybe only check car's middle point ? (can make shortcuts with the mask version)
        rotated_image = pygame.transform.rotate(self.car.image, self.car.angle)

        car_mask = pygame.mask.from_surface(rotated_image)
        track_mask = pygame.mask.from_surface(self.track.image)

        rotated_rect = rotated_image.get_rect(center=(self.car.x_position, self.car.y_position))

        return not bool(car_mask.overlap(track_mask, (0-rotated_rect.topleft[0], 0-rotated_rect.topleft[1])))


# When ran as main, the game will use player inputs.
if __name__ == '__main__':

    game = Game(CarID.F1.value, TrackID.SIMPLE.value)

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
