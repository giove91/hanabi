import sys

from game import Game

# TODO: read number of players from command line

if __name__ == "__main__":
    
    wait_key = True
    if '-c' in sys.argv[1:]:
        wait_key = False
    
    game = Game(num_players=5, wait_key=wait_key, log=True, strategy_debug=True)

    game.setup()
    game.run_game()


