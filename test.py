from game import Game

game = Game(5)

game.setup()

for player in game.players:
    print player.hand

game.run_game()
