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
    res = game.run_game()
    results.append(res)
print

print "Results"
print sorted(results)
print "Average result:", float(sum(results)) / len(results)
print "Best result:", max(results)
print "Worst result:", min(results)
print "Rate of perfect scores: %.2f\%" % float(results.count(30)) / len(results) * 100.0
