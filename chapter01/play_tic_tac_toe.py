from chapter01.tic_tac_toe import *

ai_one = RLTicTacToe(X)
ai_two = RLTicTacToe(O)

starting_player = X
N = 50000

for i in range(0, N):
    play_game(starting_player, ai_model_one=ai_one, ai_model_two=ai_two, training=True, delay=0)
    starting_player = X if starting_player == O else O

ai_one.save('data/model_X_{}k.pickle'.format(int(N/1000)))
