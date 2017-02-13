from chapter01.tic_tac_toe import *

ai_one = RLTicTacToe(X, greedy_factor=0.75)
ai_two = RLTicTacToe(O)
game = TicTacToeGame(ai_one, ai_two)

starting_player = X
N = 10000

for i in range(0, N):
    game.play(starting_player, training=True)
    starting_player = X if starting_player == O else O

ai_one.dump_json_model('data/easy_model.json')
ai_one.save('data/easy_model.pickle')

N = 15000

for i in range(0, N):
    game.play(starting_player, training=True)
    starting_player = X if starting_player == O else O

ai_one.dump_json_model('data/medium_model.json')
ai_one.save('data/medium_model.pickle')

N = 100000

for i in range(0, N):
    game.play(starting_player, training=True)
    starting_player = X if starting_player == O else O

ai_one.dump_json_model('data/hard_model.json')
ai_one.save('data/hard_model.pickle')

