#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script checks that the current result of alphahanabi matches the result of the challenge played on 21/07/2016.
"""


import sys

from game import Game

SCORES = {
    0: 27,
    1: 26,
    2: 27,
    3: 29,
}

for (i, score) in SCORES.iteritems():
    game = Game(
            num_players=5,
            ai="alphahanabi",
            wait_key=False,
            log=False,
            strategy_log=False,
            dump_deck_to=None,
            load_deck_from="challenge-2016-07-21/game%d.txt" % i,
        )
    
    game.setup()
    statistics = game.run_game()
    
    print "Game %d" % i
    print statistics
    assert statistics.score == score

