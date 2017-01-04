import numpy as np
from math import floor

ROWS = 3
COLS = 3

EMPTY = 0
X = 1
O = 2


def invert_state(state):
    result = list(state)
    for tile, type in enumerate(result):
        if type == EMPTY:
            pass
        else:
            result[tile] = (type % O) + 1

    return result


def generate_state_space():
    state_space = dict()
    root_state = np.ones((ROWS, COLS)).astype(int) * EMPTY
    root_state_key = state_to_string(root_state)
    state_space[root_state_key] = 0.5

    return state_space


def state_to_string(state):
    return ''.join(map(str, state.flatten().tolist()))


def string_to_state(state):
    if len(state) != 9:
        raise ValueError('The input state must be 9 digits long.')
    state_list = list(map(int, list(state)))
    return np.array([state_list[:3], state_list[3:6], state_list[6:]])


# Use numpy array to store the board state and then use this function to convert between a range index and coordinates.
def index_to_board_coordinates(index):
    if index < 0 or index > 8:
        raise IndexError('Index out of bounds. Index is bounded by 0 and 8')
    row = floor(index / COLS)
    col = index % 3

    return row, col
