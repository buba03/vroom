""" Module for yaml management. """

import yaml

from src.utils.enums import CarID


class YamlManager:
    """ Class for yaml management. """
    def __init__(self, path: str):
        """
        Sets up the yaml manager with the given path.

        :param path: Path to the yaml file.
        """
        self._path = path
        self.values = self._load_yaml(self._path)

    @staticmethod
    def _load_yaml(path: str) -> dict:
        """
        Loads the yaml file.

        :param path: Path to the yaml file to load.
        :return: The contents of the loaded yaml as a dictionary.
        """
        with open(path, 'r', encoding="utf-8") as file:
            return yaml.safe_load(file)

    def get_car_attributes(self, car_id: CarID) -> dict:
        """
        Gets the car's attributes. Calculates the percentage based values.

        :param car_id: The ID of the car to get the attributes for.
        :return: The car's attributes as a dictionary.
        """
        normalized_attributes = {}

        for attribute in self.values[car_id]:
            max_value = self.values['max_values'][attribute]
            min_value = self.values['min_values'][attribute]
            value = self.values[car_id][attribute]

            normalized_attributes[attribute] = (max_value - min_value) * (value / 100) + min_value

        return normalized_attributes
