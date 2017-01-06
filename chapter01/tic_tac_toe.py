import numpy as np
from math import floor

ROWS = 3
COLS = 3

EMPTY = 0
X = 1
O = 5

START_PROB = 0.5  #The default probability estimate of winning is 50%


def invert_state(state):
    result = list(map(int, list(state)))

    for tile, player in enumerate(result):
        if player == EMPTY:
            pass
        else:
            result[tile] = O if player == X else X

    return list_to_string(result)


def generate_state_space():
    state_space = dict()
    root_state = [0] * (ROWS * COLS)
    root_state_key = list_to_string(root_state)
    state_space[root_state_key] = START_PROB

    add_states(root_state, X, state_space)

    for state in list(state_space.keys()):
        inverted_state = invert_state(state)
        state_space[inverted_state] = START_PROB

    return state_space


def add_states(root_state, player, state_space):
    """
    Go through all the tiles on the board. If there is an empty tile, fill it with the current player's symbol,
    change the player and do repeat with the new board state until the entire board has been filled.
    This will recursively fill the state_space from the given root_state.

    :param root_state: a list of length 9 with each element representing a tile on the TicTacToe board.
    :param player: an int: 1 or 2 for X or O
    :param state_space: a dict containing the current game states as keys
    :return: Nothing. Adds new elements to the state_space dict
    """

    for i, tile in enumerate(root_state):
        if tile == EMPTY:
            new_state = list(root_state)
            new_state[i] = player
            new_state_key = ''.join(map(str, new_state))
            state_space[new_state_key] = START_PROB

            next_player = O if player == X else X
            # Continue down the game tree
            add_states(new_state, next_player, state_space)


def list_to_string(a_list):
    return ''.join(map(str, a_list))


def state_to_string(state):
    return ''.join(map(str, state.flatten().tolist()))


def string_to_state(state):
    if len(state) != 9:
        raise ValueError('The input state must be 9 digits long.')
    state_list = list(map(int, list(state)))
    return np.array([state_list[:3], state_list[3:6], state_list[6:]])


# Use numpy array to store the board state and then use this function to convert between a range index and coordinates.
# Only use it for active games.
def index_to_board_coordinates(index):
    if index < 0 or index > 8:
        raise IndexError('Index out of bounds. Index is bounded by 0 and 8')
    row = floor(index / COLS)
    col = index % 3

    return row, col


