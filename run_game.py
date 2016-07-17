import sys

from game import Game

# TODO: read number of players from command line

if __name__ == "__main__":
    # default values
    wait_key = True
    num_players = 4
    log = True
    strategy_debug = False
    
    # read values
    if '-c' in sys.argv[1:]:
        wait_key = False
    
    if '-d' in sys.argv[1:]:
        strategy_debug = True
    
    
    
    # run game
    print "Starting game with %d players..." % num_players
    print
    game = Game(num_players=num_players, wait_key=wait_key, log=log, strategy_debug=strategy_debug)

    game.setup()
    statistics = game.run_game()
    
    print statistics
