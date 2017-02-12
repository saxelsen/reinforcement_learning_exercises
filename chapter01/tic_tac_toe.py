import numpy as np
from math import floor
import numpy.random as nprandom
import random
import time
import pickle

seed = 100
nprandom.seed(seed)
random.seed(seed)

ROWS = 3
COLS = 3

#TODO: Wrap these in an Enum class
EMPTY = '-'
X = 'X'
O = 'O'

START_PROB = 0.5  #The default probability estimate of winning is 50%
EMPTY_BOARD = np.array([EMPTY] * 9).reshape((ROWS, COLS))

class TicTacToeUtils:

    @staticmethod
    def is_winning_state_for_player(state, player):
        winner = False

        if state[0] == player and state[1] == player and state[2] == player:
            winner = True
        elif state[3] == player and state[4] == player and state[5] == player:
            winner = True
        elif state[6] == player and state[7] == player and state[8] == player:
            winner = True
        elif state[0] == player and state[3] == player and state[6] == player:
            winner = True
        elif state[1] == player and state[4] == player and state[7] == player:
            winner = True
        elif state[2] == player and state[5] == player and state[8] == player:
            winner = True
        elif state[0] == player and state[4] == player and state[8] == player:
            winner = True
        elif state[2] == player and state[4] == player and state[6] == player:
            winner = True

        return winner

    @staticmethod
    def list_to_string(a_list):
        return ''.join(a_list)

    @staticmethod
    def state_to_string(state):
        return ''.join(state.flatten().tolist())

    @staticmethod
    def string_to_state(state):
        if len(state) != 9:
            raise ValueError('The input state must be 9 digits long.')
        state_list = list(state)
        return np.array([state_list[:3], state_list[3:6], state_list[6:]])

    @staticmethod
    def index_to_board_coordinates(index):
        if index < 0 or index > 8:
            raise IndexError('Index out of bounds. Index is bounded by 0 and 8')
        row = floor(index / COLS)
        col = index % 3

        return row, col

    @staticmethod
    def print_board(board):
        player_to_string = {EMPTY: ' ', X: 'X', O: 'O'}

        for row in board:
            row_string = '|'
            for col in row:
                row_string += player_to_string[col] + '|'
            print(row_string)
        print('')


class TicTacToeGenerator:

    def __init__(self, origin_player_symbol, opponent_symbol):
        self.origin_player = origin_player_symbol
        self.opponent = opponent_symbol

    def invert_state(self, state):
        result = list(state)

        for tile, player in enumerate(result):
            if player == EMPTY:
                pass
            else:
                result[tile] = self.opponent if player == self.origin_player else self.origin_player

        return TicTacToeUtils.list_to_string(result)

    def revert_state_space(self, space):
        #TODO: Should revert_state_space invert the keys or invert the probabilities???
        result = dict()

        for state_key, probability in space.items():
            inverse_state_key = self.invert_state(state_key)
            result[inverse_state_key] = probability

        return result

    def generate_state_space(self):
        state_space = dict()
        root_state = EMPTY_BOARD.flatten().tolist()
        root_state_key = TicTacToeUtils.list_to_string(root_state)
        state_space[root_state_key] = self.default_probability(root_state_key)

        self.add_states(root_state, self.origin_player, state_space)

        for state_key in list(state_space.keys()):
            inverted_state_key = self.invert_state(state_key)
            state_space[inverted_state_key] = self.default_probability(inverted_state_key)

        return state_space

    def add_states(self, root_state, player, state_space):
        """
        Go through all the tiles on the board. If there is an empty tile, fill it with the current player's symbol,
        change the player and do repeat with the new board state until the entire board has been filled.
        This will recursively fill the state_space from the given root_state.

        :param root_state: a list of length 9 with each element representing a tile on the TicTacToe board.
        :param origin_player: an int: 1 or 2 for X or O
        :param state_space: a dict containing the current game states as keys
        :return: Nothing. Adds new elements to the state_space dict
        """

        for i, tile in enumerate(root_state):
            if tile == EMPTY:
                new_state = list(root_state)
                new_state[i] = player
                new_state_key = ''.join(new_state)
                default_prob = self.default_probability(new_state_key)
                state_space[new_state_key] = default_prob

                if 0 < default_prob < 1:
                    next_player = self.opponent if player == self.origin_player else self.origin_player
                    # Continue down the game tree if this game was not ended
                    self.add_states(new_state, next_player, state_space)

    def default_probability(self, state_string):
        default_prob = START_PROB
        state = list(state_string)

        player_wins = TicTacToeUtils.is_winning_state_for_player(state, self.origin_player)
        opponent_wins = TicTacToeUtils.is_winning_state_for_player(state, self.opponent)

        if player_wins:
            default_prob = 1
        elif opponent_wins:
            default_prob = 0

        return default_prob


class RLTicTacToe:

    model_symbol = 'M'
    enemy_symbol = 'E'

    def __init__(self, symbol, model=None, greedy_factor=0.9, learning_rate=0.5, training=False):
        self.set_symbol(symbol)
        self.greedy_factor = greedy_factor
        self.learning_rate = learning_rate
        self.generator = TicTacToeGenerator(self.model_symbol, self.enemy_symbol)
        self.training = training

        if model is None:
            self.model = self.generator.generate_state_space()
        else:
            self.model = model

    def set_symbol(self, symbol):
        self.symbol = symbol
        self.opponent = X if symbol == O else O

    def move(self, board):

        internal_board_state_list = self._convert_state_to_internal_state_list(board)

        if EMPTY not in internal_board_state_list:
            raise EnvironmentError('Cannot make a move. There are no empty tiles on the board.')

        possible_plays = dict()

        for i, tile in enumerate(internal_board_state_list):
            if tile == EMPTY:
                new_state = list(internal_board_state_list)
                new_state[i] = 'M'
                new_state_key = TicTacToeUtils.list_to_string(new_state)
                possible_plays[i] = self.model[new_state_key]

        sorted_possible_plays = sorted(possible_plays.items(), key=lambda x: x[1], reverse=True)

        greedy_move, exploratory_moves = sorted_possible_plays[0], sorted_possible_plays[1:]

        if self.training:
            is_greedy = nprandom.rand() < self.greedy_factor or len(exploratory_moves) == 0
        else:
            is_greedy = True

        if is_greedy:
            result_move = TicTacToeUtils.index_to_board_coordinates(greedy_move[0])
        else:
            exploratory_move = random.choice(exploratory_moves)
            result_move = TicTacToeUtils.index_to_board_coordinates(exploratory_move[0])

        return result_move

    def _convert_state_to_internal_state_list(self, state):
        state_list = state.flatten().tolist()
        return [var.replace(self.symbol, self.model_symbol).replace(self.opponent, self.enemy_symbol) for var in state_list]

    def update(self, old_board, new_board):
        old_internal_board_list = self._convert_state_to_internal_state_list(old_board)
        old_board_key = ''.join(old_internal_board_list)

        new_internal_board_list = self._convert_state_to_internal_state_list(new_board)
        new_board_key = ''.join(new_internal_board_list)

        new_board_value = self.model[new_board_key]
        old_board_value = self.model[old_board_key]

        self.model[old_board_key] = old_board_value + self.learning_rate * (new_board_value - old_board_value)

    def save(self, path):
        info_to_save = {'symbol': self.symbol,
                        'model': self.model,
                        'greedy_factor': self.greedy_factor,
                        'learning_rate': self.learning_rate}
        with open(path, 'wb') as file:
            pickle.dump(info_to_save, file)
        print('Model saved.')

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            info = pickle.load(file)
        ai = RLTicTacToe(info['symbol'], info['model'], info['greedy_factor'], info['learning_rate'])
        return ai


def play_game(starting_player, ai_model_one, ai_model_two=None, training=False, delay=0.5):
    """
    Play through a game. X always starts the game.
    :param starting_player: An int equivalent to X or O. If ai_player is O, the user plays X and vice-versa.
    :param ai_model_one: The AI model to play against.
    :param ai_model_two: A secondary AI model that can play the game. If this is not None, the two AIs will play each
    other. Should be used for training models.
    :return:
    """

    board = EMPTY_BOARD.copy()
    is_game_won = is_game_finished = False
    current_player = starting_player

    ai_model_one.training = training

    if ai_model_two:
        ai_model_two.training = training

    if not training:
        TicTacToeUtils.print_board(board)

    while is_game_finished is not True:

        if current_player == ai_model_one.symbol:
            move = ai_model_one.move(board)
            if not training:
                print('AI 1 played move {}'.format(move))
        else:
            if ai_model_two is None:
                move = get_user_input(board)
            else:
                move = ai_model_two.move(board)
                if not training:
                    print('AI 2 played move {}'.format(move))

        board_before = board.copy()
        board[move] = current_player

        if training:
            # Update the AIs if in training mode
            ai_model_one.update(board_before, board)
            if ai_model_two is not None:
                ai_model_two.update(board_before, board)
        else:
            TicTacToeUtils.print_board(board)

        board_state = board.flatten().tolist()
        is_game_won = TicTacToeUtils.is_winning_state_for_player(board_state, current_player)
        is_game_finished = is_game_won or (EMPTY not in board_state)

        if not is_game_finished:
            current_player = O if current_player == X else X
        time.sleep(delay)

    if is_game_won:
        player_string = current_player
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

