#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random

sys.path.append("..") 


from action import Action, PlayAction, DiscardAction, HintAction
from card import Card, deck



class Strategy:
    """
    An instance of this class represents a player's strategy.
    It only has the knowledge of that player, and it must make decisions.
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
        pass
    
    
    def get_turn_action(self):
        
        if self.hints > 0 and random.randint(0,2) == 0:
            # give random hint to the next player
            player_id = (self.id + 1) % self.num_players
            card = random.choice([card for card in self.hands[player_id] if card is not None])
            
            if random.randint(0,1) == 0:
                color = card.color
                number = None
            else:
                color = None
                number = card.number
            
            return HintAction(player_id, color=color, number=number)
        
        elif random.randint(0,1) == 0:
            # play random card
            card_pos = random.choice([c_pos for (c_pos, value) in enumerate(self.my_hand) if value is not None])
            return PlayAction(card_pos)
        
        else:
            # discard random card
            card_pos = random.choice([c_pos for (c_pos, value) in enumerate(self.my_hand) if value is not None])
            return DiscardAction(card_pos)


    def log(self, message):
        if self.verbose:
            print "Player %d: %s" % (self.id, message)




