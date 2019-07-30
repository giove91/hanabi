#!/usr/bin/env python
# -*- coding: utf-8 -*-


import multiprocessing
import sys

from game.game import Game


if __name__ == "__main__":
    # default values
    ai = "alphahanabi"
    ai_params = {}
    num_players = 5
    num_simulations = 1000
    
    
    if '-a' in sys.argv[1:]:
        # select AI to be used
        i = sys.argv.index('-a')
        assert len(sys.argv) >= i+2
        ai = sys.argv[i+1]
    
    if '-n' in sys.argv[1:]:
        # read number of players
        i = sys.argv.index('-n')
        assert len(sys.argv) >= i+2
        num_players = int(sys.argv[i+1])
    
    if '-m' in sys.argv[1:]:
        # read number of simulations
        i = sys.argv.index('-m')
        assert len(sys.argv) >= i+2
        num_simulations = int(sys.argv[i+1])
    
    if '-p' in sys.argv[1:]:
        # set difficulty parameter
        i = sys.argv.index('-p')
        assert len(sys.argv) >= i+2
        ai_params['difficulty'] = sys.argv[i+1]
    
    results = []

    print "Starting %d simulations with %d players..." % (num_simulations, num_players)
    def run_game(i):
        #print(i, end=' ', file=sys.stderr, flush = True)
        game = Game(
                num_players=num_players,
                ai=ai,
                ai_params=ai_params,
                strategy_log=False,
                dump_deck_to='deck.txt',
                load_deck_from=None,
            )
        
        game.setup()
        for current_player, turn in game.run_game():
            pass
        return game.statistics
    pool = multiprocessing.Pool(8)
    #pool.map = map # uncomment for debugging purposes
    results = pool.map(run_game, range(num_simulations))
    print

    scores = [statistics.score for statistics in results]

    print "Results"
    print sorted(scores)
    print "Number of players:", num_players
    print "Average result:", float(sum(scores)) / len(scores)
    print "Best result:", max(scores)
    print "Worst result:", min(scores)
    print "Rate of perfect scores: %.2f %%" % (float(scores.count(30)) / len(scores) * 100.0)

    lives = [statistics.lives for statistics in results]
    print "Average number of remaining lives:", float(sum(lives)) / len(lives)

    num_turns = [statistics.num_turns for statistics in results]
    print "Average number of turns:", float(sum(num_turns)) / len(num_turns)

