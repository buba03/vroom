""" Module for game logic. When ran as main, the game will use the player controls. """

import math

import pygame

from car import Car
from track import Track
from utils.enums import Direction, Color
from utils.config_manager import ConfigManager

# import random
# import numpy as np

# Initialize pygame
pygame.init()
# Constants
FONT = pygame.font.SysFont('arial', 18)


class GameAction:
    """ Helper class for handling game actions. """

    # Constants
    NO_ACTION = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    FORWARD = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    FORWARD_RIGHT = [0, 0, 1, 0, 0, 0, 0, 0, 0]
    TURN_RIGHT = [0, 0, 0, 1, 0, 0, 0, 0, 0]
    BACKWARD_RIGHT = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    BACKWARD = [0, 0, 0, 0, 0, 1, 0, 0, 0]
    BACKWARD_LEFT = [0, 0, 0, 0, 0, 0, 1, 0, 0]
    TURN_LEFT = [0, 0, 0, 0, 0, 0, 0, 1, 0]
    FORWARD_LEFT = [0, 0, 0, 0, 0, 0, 0, 0, 1]

    def __init__(self, action=-1, action_count: int = 9):
        """
        Initializes a GameAction object.

        :param action: The action stored as a list of integers. The list is filled with zeros, only one int is 1.
        Can be given as a list - the length of the list is the action_count.
        Can be given as an integer meaning the index of the list, which should be a 1.
        Default GameAction is a list of zeros.
        :param action_count: The number of possible actions. Default is 9.
        """
        # TODO into yaml
        self.action_count = action_count

        self.action = action

    @property
    def action(self) -> list:
        """
        Gets the value of the private attribute.
        """
        return self._action

    @action.setter
    def action(self, other):
        """
        Sets the value of the private attribute.
        Can be given as a list - the length of the list is the action_count.
        Can be given as an integer - meaning the index of the list, which should be a 1.
        """
        # List
        if isinstance(other, list):
            # Check length
            if len(other) != self.action_count:
                raise IndexError(f'The length of the list should be {self.action_count}, instead of {len(other)}.')
            self._action = other
        # Integer
        elif isinstance(other, int):
            # Out of bounds
            if other >= self.action_count:
                raise IndexError(
                    f'The index cannot exceed the length of the possible actions. Maximum index is {self.action_count - 1}, got: {other}.')
            # Empty action
            if other < 0:
                self._action = [0] * self.action_count
            # Action by index
            else:
                self._action = [0] * self.action_count
                self._action[other] = 1
        # Anything else
        else:
            raise TypeError('The new_action should be a list or int.')

    @property
    def action_count(self) -> int:
        """
        Gets the value of the private attribute.
        """
        return self._action_count

    @action_count.setter
    def action_count(self, other: int):
        """
        Sets the value of the private attribute.
        Only integers are allowed.
        """
        if not isinstance(other, int):
            raise TypeError(f'int expected, got {type(other)} instead.')

        self._action_count = other


class Game:
    """ Class for game logic. """

    def __init__(self, car: str, track: str):
        """
        Initializes the game.

        :param car: The name of the car inside the cars.yaml file.
        :param track: The name of the track inside the tracks.yaml file.
        """
        # yaml import
        display_attributes = ConfigManager().get_game_attributes()
        # FPS
        self.fps = int(ConfigManager().get_argument('fps'))

        # For storing progress
        self.reached_checkpoints = set()
        self.lap_count = None

        # Car
        self.car = Car(car)
        # Track
        self.track = Track(track)

        # init display
        self.display = pygame.display.set_mode((display_attributes['width'], display_attributes['height']))
        pygame.display.set_caption('Vroom v1.0.0')
        # Clock
        self.clock = pygame.time.Clock()
        # Set up elements
        self.size = ConfigManager().get_car_size_from_track(self.track.id)
        self.car.resize(self.size)
        self.reset()

    def reset(self):
        """ Sets up the default game, resets the game state. """
        x, y, angle = self.track.get_car_default_state()
        self.car.reset(x, y, angle)

        self.reached_checkpoints = set()
        self.lap_count = 0

    def play_step(self, action: list[int]) -> tuple[float, bool, float]:
        """
        Play the next step of the game using the given action.

        :param action: A GameAction object.
        :return: reward (for the executed action), done (whether the continues), score (the current score)
        """
        # Get state before applying the action
        score = self.get_score()
        reward = 0

        # Apply action
        self.apply_action_on_car(action)
        # Update progression
        self.__check_progression()

        # Game over?
        done = self.__car_offtrack()

        if self.get_score() > score:
            reward += 10

        if done:
            reward = -10

        # Event handler
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # TODO: plots?

        # Apply changes
        self.__update_display()
        self.clock.tick(self.fps)
        return reward, done, score

    def __update_display(self):
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
        self.display.blit(text, [0, 20])
        text = FONT.render(f"Position: x: {str(int(self.car.x_position))} y: {str(int(self.car.y_position))}", True,
                           Color.WHITE.value)
        self.display.blit(text, [0, 40])
        text = FONT.render(f"FPS: {str(self.clock.get_fps())}", True, Color.WHITE.value)
        self.display.blit(text, [0, 60])
        text = FONT.render(f"Progress: {str(self.reached_checkpoints)}", True, Color.WHITE.value)
        self.display.blit(text, [0, 80])
        text = FONT.render(f"Laps: {str(self.lap_count)}", True, Color.WHITE.value)
        self.display.blit(text, [0, 100])
        text = FONT.render(f"Score: {str(self.get_score())}", True, Color.WHITE.value)
        self.display.blit(text, [0, 180])

        # Update
        pygame.display.flip()

    def apply_action_on_car(self, action):
        """
        Executes the given action on the car.

        :param action: The action to execute.
        Must be a list with the length of possible action and filled with zeros, except at the index of the chosen action.
        """
        # Action handler
        if action == GameAction.FORWARD:
            self.car.accelerate()
        elif action == GameAction.FORWARD_RIGHT:
            self.car.accelerate()
            self.car.turn(Direction.RIGHT)
        elif action == GameAction.TURN_RIGHT:
            self.car.turn(Direction.RIGHT)
        elif action == GameAction.BACKWARD_RIGHT:
            self.car.brake()
            self.car.turn(Direction.RIGHT)
        elif action == GameAction.BACKWARD:
            self.car.brake()
        elif action == GameAction.BACKWARD_LEFT:
            self.car.brake()
            self.car.turn(Direction.LEFT)
        elif action == GameAction.TURN_LEFT:
            self.car.turn(Direction.LEFT)
        elif action == GameAction.FORWARD_LEFT:
            self.car.accelerate()
            self.car.turn(Direction.LEFT)
        else:
            pass

        # Make the action
        self.car.move()

    def __car_offtrack(self) -> bool:
        """
        Checks whether the track and the car has collided.

        :return: True if the car has completely left the track, False otherwise.
        """
        corners = self.car.get_corners(5)

        wheels_on_track = 0

        for position in corners:
            try:
                if self.display.get_at(position) == Color.RED.value:
                    print("REEEED")
            except IndexError:
                pass
            try:
                if self.display.get_at(position) == Color.TRACK.value:
                    wheels_on_track += 1
            # Ignore when a corner is out of the screen
            except IndexError:
                pass

        return wheels_on_track == 0

    def __get_checkpoint_reached(self) -> int:
        """
        Checks if the car reached a new checkpoint and returns it.

        :return: The index of the reached checkpoint. -1 if none is reached.
        """
        threshold = 100 * self.size * self.size

        checkpoints = self.track.checkpoints
        car_position = int(self.car.x_position), int(self.car.y_position)

        for i, checkpoint in enumerate(checkpoints):
            # FIXME debug
            # pygame.draw.circle(self.display, Color.RED.value, checkpoint, threshold, width=3)
            # pygame.draw.rect(self.display, Color.RED.value, pygame.rect.Rect(checkpoint[0] - threshold, checkpoint[1] - threshold, threshold*2, threshold*2))

            # Check threshold (square)
            if abs(checkpoint[0] - car_position[0]) < threshold and abs(checkpoint[1] - car_position[1]) < threshold:
                return i

        return -1

    def get_next_checkpoint(self) -> int:
        """
        Returns the index of next checkpoint, that should be reached.

        :return: The index of the checkpoint.
        """
        return len(self.reached_checkpoints) % len(self.track.checkpoints)

    def __check_progression(self):
        """ Checks the car's progression on the track. Updates reached checkpoints and lap counts. """

        # Get the currently touched checkpoint
        checkpoint = self.__get_checkpoint_reached()

        # Check whether it's the next one
        if checkpoint != self.get_next_checkpoint():
            # Disallow it, if going the wrong direction
            checkpoint = -1

        # If the car touched a checkpoint
        if checkpoint != -1:
            # Ignore checkpoint if the set is empty and it isn't the first checkpoint
            if len(self.reached_checkpoints) == 0 and checkpoint != 0:
                return
            # Add
            self.reached_checkpoints.add(checkpoint)
            # Check lap progress
            if len(self.reached_checkpoints) == 4 and checkpoint == 0:
                # Reset if completed a lap
                self.reached_checkpoints = set()
                self.lap_count += 1

    def __cast_ray(self, angle_offset: int = 0) -> float:
        """
        Casts a ray from the middle of the car in the given angle. The ray stops when it hits the side of the track.

        :param angle_offset: The offset of the ray's angle. Default is 0.
        :return: The length of the ray.
        """
        length = 0
        step = 1
        # Convert angle to radians and clockwise
        angle = math.radians(-(self.car.angle - angle_offset))

        # Calculate length, max value: display's width + height
        while length < self.display.get_size()[0] + self.display.get_size()[1]:
            # Gradually increase the length
            length += step
            # Coordinates of the end of the ray
            x = int(self.car.x_position + length * math.cos(angle))
            y = int(self.car.y_position + length * math.sin(angle))
            # Check if the ray hit the side of the track
            if self.track.image.get_at((x, y)) != Color.TRACK.value:
                # FIXME debug
                # pygame.draw.line(self.display, (255, 0, 0), (self.car.x_position, self.car.y_position), (x, y))
                break

        return length

    def get_rays(self) -> list[float]:
        """
        Casts all the rays and calculates their lengths.

        :return: List of the calculated distances counterclockwise.
        """
        angles = [-90, -45, 0, 45, 90]
        result = []

        for angle in angles:
            result.append(self.__cast_ray(angle))

        return result

    def get_score(self) -> int:
        """ Returns the sum of checkpoints reached in correct order. """
        return len(self.reached_checkpoints) + self.lap_count * len(self.track.checkpoints)


# When ran as main, the game will use player inputs.
if __name__ == '__main__':

    car_arg = ConfigManager().get_argument('car')
    track_arg = ConfigManager().get_argument('track')
    game = Game(car_arg, track_arg)

    # Game loop
    while True:
        # Default action (do nothing)
        player_action = GameAction.NO_ACTION

        # Currently pressed keys
        keys = pygame.key.get_pressed()

        # FIXME debug
        # if keys[pygame.K_SPACE]:
        #     print(pygame.mouse.get_pos())

        # WASD and arrow key presses to actions
        accelerate = keys[pygame.K_w] or keys[pygame.K_UP]
        brake = keys[pygame.K_s] or keys[pygame.K_DOWN]
        turn_left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        turn_right = keys[pygame.K_d] or keys[pygame.K_RIGHT]

        # Calculate action
        if accelerate and brake:
            player_action = GameAction.NO_ACTION
        elif accelerate and ((not turn_left and not turn_right) or (turn_left and turn_right)):
            player_action = GameAction.FORWARD
        elif accelerate and turn_right and not turn_left:
            player_action = GameAction.FORWARD_RIGHT
        elif turn_right and (not accelerate and not brake):
            player_action = GameAction.TURN_RIGHT
        elif brake and turn_right and not turn_left:
            player_action = GameAction.BACKWARD_RIGHT
        elif brake and (not turn_left and not turn_right):
            player_action = GameAction.BACKWARD
        elif brake and turn_left and not turn_right:
            player_action = GameAction.BACKWARD_LEFT
        elif turn_left and (not accelerate and not brake):
            player_action = GameAction.TURN_LEFT
        elif accelerate and turn_left and not turn_right:
            player_action = GameAction.FORWARD_LEFT

        # Execute action, go to next state
        _, game_over, _ = game.play_step(player_action)

        if game_over:
            game.reset()
