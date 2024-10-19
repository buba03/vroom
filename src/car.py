import pygame
import os
import numpy as np

from utils.enums import *
from utils.yaml_manager import YamlManager


def rotate_vector(vector, angle_degrees):
    angle_degrees = -angle_degrees
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    result = np.dot(rotation_matrix, vector)
    return np.round(result, decimals=10)


class Car:
    def __init__(self, car_id):
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
        self.image = self._set_image(os.path.join('resources', car_id + '.png'))

        # Position
        # TODO default position
        self.x_position = 100
        self.y_position = 100

        # To calculate movement
        self.velocity = 0
        self.angle = 0

    @staticmethod
    def _set_image(path):
        img = pygame.image.load(path)
        return pygame.transform.scale(img, (16 * 4, 8 * 4))

    def get_position(self):
        return self.x_position, self.y_position

    def _set_position(self, x=None, y=None):
        if x is not None:
            self.x_position = x
        if y is not None:
            self.y_position = y

    def accelerate(self):
        self._set_velocity(self.acceleration)

    def brake(self):
        self._set_velocity(-self.braking)

    def _set_velocity(self, acceleration):
        if self.velocity >= 0:
            self.velocity = min(self.velocity+acceleration, self.max_speed)
        elif self.velocity < 0:
            self.velocity = max(self.velocity+acceleration, self.max_reverse_speed)

    def move(self):
        friction_multiplier = -1 if self.velocity < 0 else 1
        self.velocity = self.velocity - self.friction * friction_multiplier if not abs(self.velocity) < self.friction else 0

        direction = rotate_vector((self.velocity, 0), self.angle)
        self.x_position += direction[0]
        self.y_position += direction[1]

    # FIXME
    def turn(self, direction):
        # Change multiplier based on Right - Left
        direction_multiplier = -1 if direction == Direction.RIGHT else 1
        # Change multiplier based in Velocity
        direction_multiplier = -direction_multiplier if self.velocity < 0 else direction_multiplier

        # angle = min(self.handling, self.handling * (1 / (abs(self.velocity - (self.max_speed / 3)))))

        angle = self.handling
        self.angle += angle * direction_multiplier
        self.angle = self.angle % 360
        print(angle)

    def draw(self, display):
        self.move()

        rotated_image = pygame.transform.rotate(self.image, self.angle)
        rotated_rect = rotated_image.get_rect(center=(self.x_position, self.y_position))
        display.blit(rotated_image, rotated_rect)
        pygame.draw.rect(display, (0, 255, 0), rotated_rect, 2)
