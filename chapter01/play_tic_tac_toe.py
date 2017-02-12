from chapter01.tic_tac_toe import *

ai_one = RLTicTacToe(X)
ai_two = RLTicTacToe(O)
game = TicTacToeGame(ai_one, ai_two)

starting_player = X
N = 50000

for i in range(0, N):
    game.play(starting_player, training=True)
    starting_player = X if starting_player == O else O

ai_one.save('data/model_X_{}k_3.pickle'.format(int(N/1000)))
