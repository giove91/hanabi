#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys

from game import Game


results = []

print "Starting simulations..."
for i in xrange(1000):
    print >> sys.stderr, i,
    game = Game(num_players=5, wait_key=False, log=False, strategy_debug=False)
    
    game.setup()
    statistics = game.run_game()
    results.append(statistics)
print

scores = [statistics.score for statistics in results]

print "Results"
print sorted(scores)
print "Average result:", float(sum(scores)) / len(scores)
print "Best result:", max(scores)
print "Worst result:", min(scores)
print "Rate of perfect scores: %.2f %%" % (float(scores.count(30)) / len(scores) * 100.0)

lives = [statistics.lives for statistics in results]
print "Average number of remaining lives:", float(sum(lives)) / len(lives)
