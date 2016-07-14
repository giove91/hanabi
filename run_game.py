from game import Game

# TODO: read number of players from command line


game = Game(num_players=5, strategy_debug=True)

game.setup()
game.run_game()


