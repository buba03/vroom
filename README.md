# Vroom

This project is a deep reinforcement learning environment where an AI agent learns to drive on customizable tracks using PyTorch and Pygame. It features manual gameplay, training, and evaluation modes, and supports user-defined cars and tracks through YAML configuration. The system includes visualization, flexible command-line arguments, and modular design for easy development and extension.

# Prerequisites

* **Windows 10** or above
* Python **3.11.4**
    * `virtualenv` python package
        * to install, run: `pip install virtualenv`
* pip **24.0**

# Setup

Navigate to the **project's root**.

Create a virtual environment:

```shell
virtualenv .venv
```

Activate the environment:

```shell
.venv/Scripts/activate
```

### Install dependencies

```shell
pip install -r requirements.txt
```

#### Explanation:

* `pygame`: visualization, player controls
* `torch`, `torchvision`: machine learning
* `matplotlib` `ipython`: plots
* `pyyaml`: yaml imports

To deactivate the environment:

```shell
deactivate
```

# Features

## Game

The user can play the game with **WASD / arrow** controls.

Activate the `.venv` and run the following command from the **project's root**:

```shell
python src/game.py
```

## Agent

### Training

The agent will **start training** by playing the game.

Activate the `.venv` and run the following command from the **project's root**:

```shell
python src/agent.py
```

### Evaluation

The agent will **evaluate** an existing model.

Activate the `.venv` and run the following command from the **project's root**:

```shell
python src/agent.py --eval --modell path/to/the/model.pth
```

> **NOTE**: The path should be given from inside the project's `model` folder.

## Arguments

Some **command line arguments** can be used to customize the environment. These are all optional with default values.

More details (choices and default values) are in the `src/utils/config_manager.py` file.

* `--car <car_id>`: Select the car
* `--track <track_id>`: Select the track
* `--fps <number>`: Set the FPS
* `--model <path>`: Select an existing model (.pth file) to continue training or to start evaluation
  * the path should be given from inside the project's `model` folder.
* `--eval`: The agent will start in evaluation mode
  * a model should be set with the `--model` argument

# For developers

This section explains some key features of the project, that are only important for developing.

## New car

To add a **new car**, follow these steps:

### yaml

Add the car and its attributes to the `cars.yaml` file in the `src/resources` folder.

The values are percentage based (between the `min_values` and `max_values`).

* `0` will set it according to the `min_values`.
* `100` will set it according to the `max_values`.
* anything outside this range may give unrealistic or unexpected behaviour.

#### Format:

```yaml
<car_id>:
  default_speed: <num>
  acceleration: <num>
  braking: <num>
  handling: <num>
  max_speed: <num>
  min_speed: <num>
```

Fill the values inside the `<>` brackets.

### ConfigManager

Go to the `src/utils/config_manager.py` file and add a new choice to the `--car` argument according to your `car_id`.

### Usage

When running a python file, include the `--car` argument with the value of your `car_id`.

#### Example: `python .\src\game.py --car car_id`

## New track

To add a **new track**, follow these steps:

### Image

Add your image (png) to the `src/resources/tracks` folder.

* recommended size: the same as the game window (see `game.yaml` in the `src/resources` folder).
* color: the track's image must be the same color as shown in the `src/utils/enums.py`.

> **NOTE**: The name of the image will be used as the `track_id` in the following steps.

### yaml

Add the track and its attributes to the `tracks.yaml` file in the `src/resources` folder.

#### Format:

```yaml
<track_id>:
  size: <num>
  car_default_states:
    0:
        x: <num>
        y: <num>
        angle: <num>
    1:
        x: <num>
        y: <num>
        angle: <num>
#   ...
  checkpoints:
    0:
      x: <num>
      y: <num>
    1:
      x: <num>
      y: <num>
#   ...
```

Fill the values inside the `<>` brackets.

* `size`: The size multiplier of the car. Changes the size of the image, and the attributes accordingly.
    * default size: `1`
* `car_default_states`: A possible default position of the car when the game is reset.
* `checkpoints`: At least 2 checkpoints are required.

### ConfigManager

Go to the `src/utils/config_manager.py` file and add a new choice to the `--track` argument according to your `track_id`.

### Usage

When running a python file, include the `--track` argument with the value of your `track_id`.

#### Example: `python .\src\game.py --track track_id`

## Tests

### Static Code Analysis

The `pylint` python package is required.

#### Install

Run the following command inside the `.venv`:

```shell
pip install pylint
```

#### Run

Run the following command from the project's root:

```shell
pylint src
```
