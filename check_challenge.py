#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script checks that the current result of alphahanabi matches the result of the challenge played on 21/07/2016.
Ideally, it should be part of some test suite (maybe).
"""


import sys

from game.game import Game

SCORES = {
    0: 27,
    1: 26,
    2: 27,
    3: 29,
}

difference = 0

for (i, score) in SCORES.iteritems():
    game = Game(
            num_players=5,
            ai="alphahanabi",
            ai_params={"difficulty": "moderate"},
            strategy_log=False,
            dump_deck_to=None,
            load_deck_from="challenge-2016-07-21/game%d.txt" % i,
        )
    
    game.setup()
    for _ in game.run_game():
        pass
    statistics = game.statistics
    
    diff = statistics.score - score
    difference += diff
    
    print "Game %d" % i
    print "Old score: %d (%d)" % (score, diff)
    print statistics


print
print "Difference: %d" % difference

