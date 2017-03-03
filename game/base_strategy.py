#!/usr/bin/env python
# -*- coding: utf-8 -*-


class BaseStrategy(object):
    """
    Subclass this class once for each AI.
    """
    
    def __init__(self, verbose=False, params={}):
        self.verbose = verbose
    
    
    def initialize(self, id, num_players, k, board, deck_type, my_hand, hands, discard_pile):
        """
        To be called once before the beginning.
        """
        self.id = id
        self.num_players = num_players
        self.k = k  # number of cards per hand
        self.board = board
        self.deck_type = deck_type
        
        self.my_hand = my_hand  # says in which positions there is actually a card
        self.hands = hands
        self.discard_pile = discard_pile
    
    
    def update(self, hints, lives, my_hand, hands, discard_pile, turn, last_turn, deck_size):
        """
        To be called immediately after every turn.
        """
        self.hints = hints
        self.lives = lives
        self.turn = turn
        self.last_turn = last_turn
        self.deck_size = deck_size
        
        self.my_hand = my_hand  # says in which positions there is actually a card
        self.hands = hands
        self.discard_pile = discard_pile
    
    
    def feed_turn(self, player_id, action):
        """
        Receive information about a played turn.
        """
        raise NotImplementedError
    
    
    def get_turn_action(self):
        """
        Choose action for this turn.
        """
        raise NotImplementedError


    def log(self, message):
        if self.verbose:
            print "Player %d: %s" % (self.id, message)


