""" Module for yaml management. """

import yaml
import os

PATH = os.path.join('src', 'resources')


class ConfigManager:
    """ Class for yaml management. """

    def __init__(self):
        """
        Initializes the yaml manager with the given path.

        """
        self._path = PATH
        self.values = None

    def _load_yaml(self, path: str):
        """
        Loads the yaml file. Sets self.values according to the contents of the yaml file.

        :param path: Path to the yaml file to load. (excluding the resources folder's path)
        :return: The contents of the loaded yaml as a dictionary
        """
        with open(path, 'r', encoding="utf-8") as file:
            self.values = yaml.safe_load(file)

    def get_car_image_path(self, car_id: str) -> str:
        return os.path.join(self._path, 'cars', car_id + '.png')

    def get_track_image_path(self, track_id: str) -> str:
        return os.path.join(self._path, 'tracks', track_id + '.png')

    def get_car_attributes(self, car_id: str) -> dict:
        """
        Gets the car's attributes. Calculates the percentage based values.

        :param car_id: The ID of the car to get the attributes for.
        :return: The car's attributes as a dictionary.
        """
        # Load yaml
        self._load_yaml(os.path.join(self._path, 'cars.yaml'))

        normalized_attributes = {}

        for attribute in self.values[car_id]:
            max_value = self.values['max_values'][attribute]
            min_value = self.values['min_values'][attribute]
            value = self.values[car_id][attribute]

            normalized_attributes[attribute] = (max_value - min_value) * (value / 100) + min_value

        return normalized_attributes

    def get_track_attributes(self, track_id: str) -> dict:
        """
        Gets the track's attributes.

        :param track_id: The ID of the track to get the attributes for.
        :return: The track's attributes as a dictionary.
        """
        # Load yaml
        self._load_yaml(os.path.join(self._path, 'tracks.yaml'))

        return self.values[track_id]

    def get_game_attributes(self) -> tuple[int, dict]:
        """
        Gets the game's attributes.

        :return: The game's attributes as a tuple.
        """
        # Load yaml
        self._load_yaml(os.path.join(self._path, 'game.yaml'))

        return self.values['fps'], self.values['display']

    def get_car_size_from_track(self, track_id: str) -> int:
        """
        Gets the car's size for the current track.

        :param track_id: The ID of the track.
        :return: The car's recommended size for the current track.
        """
        # Load yaml
        self._load_yaml(os.path.join(self._path, 'tracks.yaml'))

        return self.values[track_id]['size']
