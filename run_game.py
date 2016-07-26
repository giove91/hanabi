import sys

from game import Game

# TODO: aggiungere opzione -h (help)

if __name__ == "__main__":
    # default values
    wait_key = True
    num_players = 5
    log = True
    strategy_log = False
    dump_deck_to = "deck.txt"
    load_deck_from = None
    
    repeat = None  # repeat until a bad result is reached
    
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
    
    if '-d' in sys.argv[1:]:
        # dump deck to file
        i = sys.argv.index('-d')
        assert len(sys.argv) >= i+2
        dump_deck_to = sys.argv[i+1]
    
    if '-n' in sys.argv[1:]:
        # read number of players
        i = sys.argv.index('-n')
        assert len(sys.argv) >= i+2
        num_players = int(sys.argv[i+1])
    
    if '-r' in sys.argv[1:]:
        # repeat until a bad score is reached
        i = sys.argv.index('-r')
        assert len(sys.argv) >= i+2
        repeat = int(sys.argv[i+1])
    
    counter = 0
    while True:
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
        counter += 1
        
        if repeat is None:
            break
        
        elif statistics.score <= repeat:
            print "Reached score <= %d after %d games" % (repeat, counter)
            break
        
        else:
            print
            print "=========================="
            print


