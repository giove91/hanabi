#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import copy

from ...action import Action, PlayAction, DiscardAction, HintAction
from ...card import Card
from ...game import Game


class HintInformation:
    """
    The information to a single player is made of two numbers:
    - the primary information (between 0 and 2*(k-1)-1);
    - the secondary information (between 0 and cards_per_player-1).
    This class implements sum and difference, which is used to encode/decode hints.
    """
    def __init__(self, k, primary, secondary):
        self.k = k
        self.cards_per_player = Game.CARDS_PER_PLAYER[k]
        
        self.primary = primary
        self.secondary = secondary
        
        assert 0 <= primary < 2*(k-1)
        assert 0 <= secondary < cards_per_player
    
    
    def __add__(self, other):
        return HintInformation(
            k = self.k,
            primary = (self.primary + other.primary) % (2 * (self.k - 1)),
            secondary = (self.secondary + other.secondary) % self.cards_per_player
        )
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return HintInformation(
            k = self.k,
            primary = (self.primary - other.primary) % (2 * (self.k - 1)),
            secondary = (self.secondary - other.secondary) % self.cards_per_player
        )



class HintsManager:
    def __init__(self, strategy):
        self.strategy = strategy
        
        # copy something from the strategy
        self.id = strategy.id
        self.num_players = strategy.num_players
        self.k = strategy.k
        self.full_deck = strategy.full_deck
        self.board = strategy.board
        self.public_knowledge = strategy.public_knowledge
    
    
    def log(self, message):
        self.strategy.log(message)
    
    
    def compute_information(self, player_id):
        """
        Compute information for the given player (different from myself).
        Can use:
        - that player's hand;
        - all public knowledge;
        - status of the game (turn number, deck size, discard pile, board, number of hints).
        """
        assert player_id != self.id
        hand = self.strategy.hands[player_id]
        knowledge = self.public_knowledge[player_id]
        
        


