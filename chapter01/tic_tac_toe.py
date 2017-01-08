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
    state_space[root_state_key] = default_probability(root_state_key)

    add_states(root_state, X, state_space)

    for state_key in list(state_space.keys()):
        inverted_state_key = invert_state(state_key)
        state_space[inverted_state_key] = default_probability(inverted_state_key)

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
            default_prob = default_probability(new_state_key)
            state_space[new_state_key] = default_prob

            if 0 < default_prob < 1:
                next_player = O if player == X else X
                # Continue down the game tree if this game was not ended
                add_states(new_state, next_player, state_space)


def default_probability(state_string):
    default_prob = 0.5
    state = list(map(int, list(state_string)))

    x_wins = is_winning_state_for_player(state, X)

    if x_wins:
        default_prob = 1
    else:
        o_wins = is_winning_state_for_player(state, O)
        if o_wins:
            default_prob = 0

    return default_prob


def is_winning_state_for_player(state, player):
    winner = False
    goal_sum = 3*player

    if sum(state[:3]) == goal_sum or \
       sum(state[3:6]) == goal_sum or \
       sum(state[6:]) == goal_sum or \
       sum([state[i] for i in [0, 3, 6]]) == goal_sum or \
       sum([state[i] for i in [1, 4, 7]]) == goal_sum or \
       sum([state[i] for i in [2, 5, 8]]) == goal_sum or \
       sum([state[i] for i in [0, 4, 8]]) == goal_sum or \
       sum([state[i] for i in [2, 4, 6]]) == goal_sum:
        winner = True

    return winner


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


