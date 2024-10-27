""" Module for a realistic car implementation. """

import os
import pygame
import numpy as np

from utils.enums import Direction
from utils.yaml_manager import YamlManager


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


def set_image(path: str, new_width=60):
    """
    Load an image and scale it based on the length of it. Keeps the aspect ratio.

    :param path: Path to the image.
    :param new_width: The length of the car.
    :return: The image.
    """
    img = pygame.image.load(path)

    return scale_image(img, new_width=new_width)


def scale_image(img, new_width: int = None, new_height: int = None):
    """
    Scale an image to the given size. If a width or a height is not provided, keeps the aspect ratio.

    :param img: The image to be scaled.
    :param new_width: The new width of the image.
    :param new_height: The new height of the image.
    :return: The scaled image.
    """
    width, height = img.get_size()
    aspect_ratio = height / width

    if new_height is None:
        new_height = int(new_width * aspect_ratio)
    if new_width is None:
        new_width = int(new_height / aspect_ratio)

    return pygame.transform.scale(img, (new_width, new_height))


class Car:
    """ Represents a controllable car. """

    def __init__(self, car_id: str):
        """
        Sets up the car object according to the car_id.

        :param car_id: The name of the car inside the cars.yaml file.
        """
        # Set ID
        self.id = car_id

        # yaml import
        car_attributes = YamlManager(os.path.join('resources', 'cars.yaml')).get_car_attributes(car_id)

        # Set values from the yaml
        self.acceleration = car_attributes['acceleration']
        self.braking = car_attributes['braking']
        self.handling = car_attributes['handling']
        self.friction = car_attributes['friction']
        self.max_speed = car_attributes['max_speed']
        self.max_reverse_speed = -car_attributes['max_reverse_speed']

        # Set car image according to the car_id
        self.image = set_image(os.path.join('resources', 'cars', car_id + '.png'))

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
        # Size
        new_width, _ = self.image.get_size()
        self.image = scale_image(self.image, new_width=(new_width*multiplier))
        # Attributes
        self.acceleration *= multiplier
        self.braking *= multiplier
        self.handling /= multiplier
        self.friction *= multiplier
        self.max_speed *= multiplier
        self.max_reverse_speed *= multiplier

    def get_center_position(self) -> tuple[float, float]:
        """ The top left position of the car. """
        return self.x_position, self.y_position

    def _set_position(self, x: float = None, y: float = None):
        """ Set the center position of the car. """
        if x is not None:
            self.x_position = x
        if y is not None:
            self.y_position = y

    def accelerate(self):
        """ Accelerate the car. Uses the car's acceleration attribute. """
        self._set_velocity(self.acceleration)

    def brake(self):
        """ Slow down or reverse the car. Uses the car's braking attribute. """
        self._set_velocity(-self.braking)

    def _set_velocity(self, acceleration: float):
        """
        Set the velocity of the car.

        :param acceleration: The acceleration of the car. | Positive: speed up | Negative: slow down / reverse
        """
        if self.velocity >= 0:
            self.velocity = min(self.velocity+acceleration, self.max_speed)
        elif self.velocity < 0:
            self.velocity = max(self.velocity+acceleration, self.max_reverse_speed)

    def move(self):
        """
        Move the car based on the current velocity and angle of the car. Applies friction.
        Uses the car's friction attribute.
        """
        # Change friction multiplier based on velocity
        friction_multiplier = -1 if self.velocity < 0 else 1
        # Apply friction
        self.velocity = self.velocity - self.friction * friction_multiplier if not abs(self.velocity) < self.friction else 0

        # Calculate movement vector based on velocity and angle
        direction = rotate_vector((self.velocity, 0), self.angle)
        # Change position based on the movement vector
        self.x_position += direction[0]
        self.y_position += direction[1]

    def turn(self, direction: Direction):
        """
        Turn the car based on its velocity and handling.
        Calculates the direction based on the forward or backward movement.
        Uses the car's handling attribute.

        :param direction: The direction of the turn.
        """
        # Change multiplier based on Right - Left
        direction_multiplier = -1 if direction == Direction.RIGHT else 1
        # Change multiplier based in Velocity
        direction_multiplier = -direction_multiplier if self.velocity < 0 else direction_multiplier

        # Slow movement speed -> slow turning speed
        angle = self.handling * abs(self.velocity)

        # Normalizing handling if getting faster
        min_threshold = 4
        max_threshold = self.max_speed
        # Normalize between 0 and 1
        normalized_velocity = 1 - (abs(self.velocity) - min_threshold) / (max_threshold - min_threshold)
        # Narrow down normalization
        min_narrowing_threshold = 0.75
        max_narrowing_threshold = 0.95
        normalized_velocity = min_narrowing_threshold + normalized_velocity * (max_narrowing_threshold - min_narrowing_threshold)

        # Change turning angle if getting faster
        angle = min_threshold * self.handling * normalized_velocity if abs(self.velocity) > min_threshold else angle

        # Apply new angle
        self.angle += angle * direction_multiplier
        self.angle = self.angle % 360

    def draw(self, surface):
        """
        Draws the car on the given surface.

        :param surface: The surface the car should be displayed on.
        """
        # Draw car based on rotation
        rotated_image = pygame.transform.rotate(self.image, self.angle)
        rotated_rect = rotated_image.get_rect(center=(self.x_position, self.y_position))
        # Draw
        surface.blit(rotated_image, rotated_rect)

        # Hit-box
        # pygame.draw.rect(surface, (0, 255, 0), rotated_rect, 2)

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
