import numpy as np
from math import floor

ROWS = 3
COLS = 3

#TODO: Wrap these in an Enum class
EMPTY = 0
X = 1
O = 5

START_PROB = 0.5  #The default probability estimate of winning is 50%
EMPTY_BOARD = np.zeros((ROWS, COLS)).astype(int)

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
    root_state = EMPTY_BOARD.flatten().tolist()
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
    default_prob = START_PROB
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


def print_board(board):
    player_to_string = {EMPTY: ' ', X: 'X', O: 'O'}

    for row in board:
        row_string = '|'
        for col in row:
            row_string += player_to_string[col] + '|'
        print(row_string)


def play_game(ai_player, ai_model_one, ai_model_two=None):
    """
    Play through a game. X always starts the game.
    :param ai_player: An int equivalent to X or O. If ai_player is O, the user plays X and vice-versa.
    :param ai_model_one: The AI model to play against.
    :param ai_model_two: A secondary AI model that can play the game. If this is not None, the two AIs will play each
    other. Should be used for training models.
    :return:
    """

    board = EMPTY_BOARD.copy()
    is_game_won = is_game_finished = False
    current_player = X
    print_board(board)

    while is_game_finished is not True:

        if current_player == ai_player:
            move = ai_model_one.move(board)
        else:
            if ai_model_two is None:
                move = get_user_input(board)
            else:
                move = ai_model_two.move(board)

        board[move] = current_player
        print_board(board)

        board_state = board.flatten().tolist()
        is_game_won = is_winning_state_for_player(board_state, current_player)
        is_game_finished = is_game_won or (EMPTY not in board_state)

    if is_game_won:
        player_string = 'X' if current_player == X else 'O'
        print('Player {} won the game'.format(player_string))
    else:
        print('The game was a tie.')


def input_to_move(user_input):
    x, y = user_input.split(',')
    return int(x), int(y)


def get_user_input(board):
    print('Please enter a move:')
    user_in = input()
    try:
        x, y = input_to_move(user_in)
    except Exception as e:
        print('Move is invalid.')
        x, y = get_user_input(board)

    if not (0 <= x <= 2) or not (0 <= y <= 2) or (board[x, y] != EMPTY):
        print('Move is invalid.')
        x, y = get_user_input(board)

    return x, y



