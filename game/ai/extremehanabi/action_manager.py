#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import copy

from ...action import Action, PlayAction, DiscardAction, HintAction
from ...card import Card

class ActionManager(object):
    def __init__(self, strategy):
        self.strategy = strategy    # my strategy object
        
        # copy something from the strategy
        self.id = strategy.id
        self.num_players = strategy.num_players
        self.k = strategy.k
        self.possibilities = strategy.possibilities
        self.full_deck = strategy.full_deck
        self.board = strategy.board
        self.knowledge = strategy.knowledge
        
        self.COLORS_TO_NUMBERS = {color: i for (i, color) in enumerate(Card.COLORS)}
    
    def log(self, message):
        self.strategy.log(message)
