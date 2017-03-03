#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script checks that the current result of alphahanabi matches the result of the challenge played on 21/07/2016.
Ideally, it should be part of some test suite (maybe).
"""


import sys

from game.game import Game

SCORES = {
    '2016-07-21': {
        0: (27, 25, 28),
        1: (26, 28, 29),
        2: (27, 28, 28),
        3: (29, 29, 30),
    },
    
    '2016-10-11': {
        0: (28, 30, 30),
        1: (28, 29, 30),
        2: (26, 27, 27),
    },
    
    '2016-10-17': {
        0: (27, 25, 30),
        1: (26, 29, 29),
        2: (27, 30, 30),
    },
    
    '2016-11-08': {
        0: (28, 29, 30),
        1: (26, 25, 29),
        2: (27, 26, 30),
    },
    
    '2016-11-15': {
        0: (28, 28, 30),
        1: (27, 28, 28),
        2: (27, 29, 29),
    },
    
    '2016-11-24': {
        0: (27, 29, 30),
    },
    
    '2016-12-11': {
        0: (26, 30, 30),
        1: (27, 29, 30),
        2: (30, 30, 30),
    },
    
    '2017-01-19': {
        0: (27, 29, 30),
        1: (26, 27, 29),
        2: (27, 25, 29),
    },
    
    '2017-01-29': {
        0: (27, 28, 30),
    },
}

DIFFICULTIES = ('moderate', 'hard', 'hardest')


difference = 0

for (date, stats) in SCORES.iteritems():
    for (i, scores) in stats.iteritems():
        for (j, score) in enumerate(scores):
            game = Game(
                num_players=5,
                ai="alphahanabi",
                ai_params={"difficulty": DIFFICULTIES[j]},
                strategy_log=False,
                dump_deck_to=None,
                load_deck_from="challenges/challenge-%s/game%d.txt" % (date, i),
            )
            
            game.setup()
            for _ in game.run_game():
                pass
            statistics = game.statistics
            
            diff = statistics.score - score
            difference += abs(diff)
            
            print "%s Game %d %s:" % (date, i, DIFFICULTIES[j])
            print "Old score: %d (%d)" % (score, diff)
            print statistics


print
print "Difference: %d" % difference

