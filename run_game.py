import sys

from game import Game

# TODO: aggiungere opzione -d per specificare su che file dumpare il mazzo
# TODO: aggiungere opzione -h (help)

if __name__ == "__main__":
    # default values
    wait_key = True
    num_players = 5
    log = True
    strategy_log = False
    dump_deck_to = "deck.txt"
    load_deck_from = None
    
    
    # read options
    if '-c' in sys.argv[1:]:
        wait_key = False
    
    if '-s' in sys.argv[1:]:
        strategy_log = True
    
    if '-l' in sys.argv[1:]:
        # load deck from file
        i = sys.argv.index('-l')
        assert len(sys.argv) >= i+2
        load_deck_from = sys.argv[i+1]
    
    if '-n' in sys.argv[1:]:
        # read number of players
        i = sys.argv.index('-n')
        assert len(sys.argv) >= i+2
        num_players = int(sys.argv[i+1])
    
    
    # run game
    print "Starting game with %d players..." % num_players
    print
    game = Game(
            num_players=num_players,
            wait_key=wait_key,
            log=log,
            strategy_log=strategy_log,
            dump_deck_to=dump_deck_to,
            load_deck_from=load_deck_from,
        )

    game.setup()
    game.log_deck()
    statistics = game.run_game()
    
    print statistics


