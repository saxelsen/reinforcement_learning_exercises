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


def next_state(state):
    result_state = list(state)
    last_tile = EMPTY
    index_of_next_empty = 0
    for i, tile in enumerate(state):
        if tile == EMPTY and last_tile != EMPTY:
            index_of_next_empty = i
            break
        else:
            last_tile = tile

    result_state[index_of_next_empty] = X
    return result_state


def generate_state_space():
    state_space = dict()
    initial_state = (EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY)


# Use numpy array to store the board state and then use this function to convert between a range index and coordinates.
def index_to_board_coordinates(index):
    if index < 0 or index > 8:
        raise IndexError("Index out of bounds. Index is bounded by 0 and 8")
    row = floor(index / COLS)
    col = index % 3

    return row, col
