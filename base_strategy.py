#!/usr/bin/env python
# -*- coding: utf-8 -*-


class BaseStrategy(object):
    """
    Subclass this class once for each AI.
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    
    def initialize(self, id, num_players, k, hands, board, discard_pile):
        # to be called once before the beginning
        self.id = id
        self.num_players = num_players
        self.k = k  # number of cards per hand
        self.hands = hands  # hands of other players
        self.board = board
        self.discard_pile = discard_pile
    
    
    def update(self, hints, lives, my_hand, turn, last_turn, deck_size):
        # to be called every turn
        self.hints = hints
        self.lives = lives
        self.my_hand = my_hand
        self.turn = turn
        self.last_turn = last_turn
        self.deck_size = deck_size
    
    
    def feed_turn(self, player_id, action):
        raise Exception("Non-overloaded method 'feed_turn'.")
    
    
    def get_turn_action(self):
        raise Exception("Non-overloaded method 'get_turn_action'.")


    def log(self, message):
        if self.verbose:
            print "Player %d: %s" % (self.id, message)


