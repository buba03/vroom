""" Module for a realistic car implementation. """

import math

import numpy as np
import pygame

from utils.config_manager import ConfigManager
from utils.enums import Color
from utils.enums import Direction

HANDLING_VELOCITY_THRESHOLD = 5


def rotate_vector(vector, angle_degrees: float):
    """
    Rotates a vector based on an angle.

    :param vector: array_like, x and y values
    :param angle_degrees: rotation angle in degrees
    :return: rotated vector
    """

    angle_degrees = -angle_degrees
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    result = np.dot(rotation_matrix, vector)
    return np.round(result, decimals=10)


def get_corner_positions(size: tuple, center: tuple, angle: float) -> list[tuple]:
    """
    Calculates the corners of a rotated rectangle.

    :param size: the size of the rectangle as a tuple (width, height).
    :param center: the center position of the rectangle as a tuple (x, y).
    :param angle: the angle of the rotation in degrees.
    :return: The rotated rectangle's corner positions as a list of tuples (x, y).
    """
    w, h = size
    cx, cy = center
    theta = -math.radians(angle)  # Negative, because math and pygame use angles differently

    # Corner positions (without rotating)
    corners = [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx - w / 2, cy + h / 2),
        (cx + w / 2, cy + h / 2)
    ]

    # Rotate the corners
    rotated_corners = []
    for x, y in corners:
        x_new = cx + (x - cx) * math.cos(theta) - (y - cy) * math.sin(theta)
        y_new = cy + (x - cx) * math.sin(theta) + (y - cy) * math.cos(theta)
        rotated_corners.append((int(x_new), int(y_new)))

    return rotated_corners


class Car:
    """ Represents a controllable car. """

    def __init__(self, car_id: str):
        """
        Initializes the car object according to the car_id.

        :param car_id: The name of the car inside the cars.yaml file.
        """
        # Set ID
        self.id = car_id

        # yaml import
        car_attributes = ConfigManager().get_car_attributes(car_id)
        self.car_width = ConfigManager().get_game_attributes()['car']['width']
        self.car_height = ConfigManager().get_game_attributes()['car']['height']

        # Set values from the yaml
        self.acceleration = car_attributes['acceleration']
        self.braking = car_attributes['braking']
        self.handling = car_attributes['handling']
        self.friction = car_attributes['friction']
        self.max_speed = car_attributes['max_speed']
        self.max_reverse_speed = -car_attributes['max_reverse_speed']

        # Set surface representing the car
        self.car_surface = pygame.Surface((self.car_width, self.car_height), pygame.SRCALPHA)
        self.car_surface.fill(Color.RED.value)

        # Position
        self.x_position = None
        self.y_position = None

        # To calculate movement
        self.velocity = 0
        self.angle = 0

    def resize(self, multiplier: int):
        """
        Resizes the car's image according to the given multiplier. Changes the car's attributes as well.

        :param multiplier: The multiplier of the resize.
        """
        global HANDLING_VELOCITY_THRESHOLD

        # Size
        self.car_surface = pygame.transform.scale(
            self.car_surface,
            (self.car_width * multiplier, self.car_height * multiplier)
        )

        # Attributes
        self.acceleration *= multiplier * multiplier
        self.braking *= multiplier * multiplier
        # self.handling /= multiplier
        self.friction *= multiplier * multiplier
        self.max_speed *= multiplier * multiplier
        self.max_reverse_speed *= multiplier * multiplier

        HANDLING_VELOCITY_THRESHOLD *= multiplier

    def get_center_position(self) -> tuple[int, int]:
        """ The top left position of the car. """
        return int(self.x_position), int(self.y_position)

    def __set_velocity(self, acceleration: float):
        """
        Set the velocity of the car.

        :param acceleration: The acceleration of the car. | Positive: speed up | Negative: slow down / reverse
        """
        if self.velocity >= 0:
            self.velocity = min(self.velocity + acceleration, self.max_speed)
        elif self.velocity < 0:
            self.velocity = max(self.velocity + acceleration, self.max_reverse_speed)

    def accelerate(self):
        """ Accelerate the car. Uses the car's acceleration attribute. """
        self.__set_velocity(self.acceleration)

    def brake(self):
        """ Slow down or reverse the car. Uses the car's braking attribute. """
        self.__set_velocity(-self.braking)

    def turn(self, direction: Direction):
        """
        Turn the car based on its velocity and handling.
        Calculates the direction based on the forward or backward movement.
        Uses the car's handling attribute.

        :param direction: The direction of the turn.
        """
        # Apply extra friction when turning
        self.apply_friction()

        # Change multiplier based on Right - Left
        direction_multiplier = -1 if direction == Direction.RIGHT else 1
        # Change multiplier based in Velocity
        direction_multiplier = -direction_multiplier if self.velocity < 0 else direction_multiplier

        # Slow movement speed -> slow turning speed
        angle = self.handling * abs(self.velocity)

        # Normalizing handling if getting faster
        min_threshold = HANDLING_VELOCITY_THRESHOLD
        max_threshold = self.max_speed
        # Normalize between 0 and 1
        normalized_velocity = 1 - (abs(self.velocity) - min_threshold) / (max_threshold - min_threshold)
        # Narrow down normalization
        min_narrowing_threshold = 0.75
        max_narrowing_threshold = 0.95
        normalized_velocity = min_narrowing_threshold + normalized_velocity * (
                max_narrowing_threshold - min_narrowing_threshold)

        # Change turning angle if getting faster
        angle = min_threshold * self.handling * normalized_velocity if abs(self.velocity) > min_threshold else angle

        # Apply new angle
        self.angle += angle * direction_multiplier
        self.angle = self.angle % 360

    def move(self):
        """
        Move the car based on the current velocity and angle of the car. Applies friction.
        Uses the car's friction attribute.
        """
        # Apply friction before moving
        self.apply_friction()

        # Calculate movement vector based on velocity and angle
        direction = rotate_vector((self.velocity, 0), self.angle)
        # Change position based on the movement vector
        self.x_position += direction[0]
        self.y_position += direction[1]

    def apply_friction(self):
        """ Subtracts the friction from the car's current velocity. Uses the car's friction attribute. """

        # Change friction multiplier based on velocity
        friction_multiplier = -1 if self.velocity < 0 else 1
        # Apply friction
        self.velocity = self.velocity - self.friction * friction_multiplier if not abs(
            self.velocity) < self.friction else 0

    def get_corners(self, offset) -> list[tuple]:
        """
        Calls the get_corner_positions() function with the car's attributes and a given offset.

        :param offset: The corners positions will be further from the center by this offset.
        :return: The rotated rectangle's corner positions as a list of tuples (x, y)
        """
        size = self.car_surface.get_size()[0] + offset, self.car_surface.get_size()[1] + offset

        return get_corner_positions(size, self.get_center_position(), self.angle)

    def draw(self, surface):
        """
        Draws the car on the given surface.

        :param surface: The surface the car should be displayed on.
        """
        # Rotate
        rotated_surface = pygame.transform.rotate(self.car_surface, self.angle)
        rotated_rect = rotated_surface.get_rect(center=(self.x_position, self.y_position))
        # Draw
        surface.blit(rotated_surface, rotated_rect.topleft)

        # FIXME debug
        # corners = self.get_corners()
        # for pos in corners:
        #     pygame.draw.circle(surface, (0, 0, 255), pos, 2)

    def reset(self, x, y, angle):
        """
        Resets the car to the given position and angle. Resets velocity.

        :param x: x position of the car
        :param y: y position of the car
        :param angle: angle of the car
        """
        self.x_position = x
        self.y_position = y
        self.angle = angle
        self.velocity = 0
