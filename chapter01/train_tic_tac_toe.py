from chapter01.tic_tac_toe import *

ai_one = RLTicTacToe(X, greedy_factor=0.75, learning_rate=1)
ai_two = RLTicTacToe(O, greedy_factor=0.9)
game = TicTacToeGame(ai_one, ai_two)

starting_player = X
match_accumulator = 0
decay = 0.99995
N = 10000

for i in range(0, N):
    game.play(starting_player, training=True)
    starting_player = X if starting_player == O else O
    match_accumulator += 1
    ai_one.learning_rate = decay**match_accumulator

ai_one.dump_json_model('data/easy_model.json')
ai_one.save('data/easy_model.pickle')

N = 15000

for i in range(0, N):
    game.play(starting_player, training=True)
    starting_player = X if starting_player == O else O
    match_accumulator += 1
    ai_one.learning_rate = decay ** match_accumulator

ai_one.dump_json_model('data/medium_model.json')
ai_one.save('data/medium_model.pickle')

N = 100000

for i in range(0, N):
    game.play(starting_player, training=True)
    starting_player = X if starting_player == O else O
    match_accumulator += 1
    ai_one.learning_rate = decay ** match_accumulator

ai_one.dump_json_model('data/hard_model.json')
ai_one.save('data/hard_model.pickle')

