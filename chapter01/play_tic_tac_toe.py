from chapter01.tic_tac_toe import *

#TODO: Currently both models have state_spaces where X-winner boards are the best. The generator should be put in the class and be specific for the model.
ai_one = RLTicTacToe.load('data/model_X.pickle')
ai_two = RLTicTacToe.load('data/model_O.pickle')
# play_game(X, ai_model_one=ai_one, ai_model_two=ai_two, training=True, delay=0)
