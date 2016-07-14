#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from termcolor import colored

from action import Action


class Card:
    RED = 'Red'
    BLUE = 'Blue'
    WHITE = 'White'
    YELLOW = 'Yellow'
    GREEN = 'Green'
    RAINBOW = 'Rainbow'
    
    NUM_COLORS = 6
    NUM_NUMBERS = 5
    
    COLORS = [RED, BLUE, WHITE, YELLOW, GREEN, RAINBOW]
    
    PRINTABLE_COLORS = {
            RED: 'red',
            BLUE: 'blue',
            WHITE: 'grey',
            YELLOW: 'yellow',
            GREEN: 'green',
            RAINBOW: 'magenta'
        }
    
    
    def __init__(self, id, color, number):
        assert color in self.COLORS
        assert 1 <= number <= 5
        
        self.id = id
        self.color = color
        self.number = number
    
    
    def __repr__(self):
        return colored("%d %s" % (self.number, self.color), self.PRINTABLE_COLORS[self.color])
        # return colored("%d" % self.number, self.PRINTABLE_COLORS[self.color])

    
    def __hash__(self):
        return self.id

    
    def __eq__(self, other):
        return self.id == other.id


    def __ne__(self, other):
        return self != other
    
    def equals(self, other):
        # same color and number (but possibly different id)
        return self.color == other.color and self.number == other.number
    
    
    def matches(self, color=None, number=None):
        # does this card match the given color/number?
        return color == self.color or number == self.number
    
    def matches_hint(self, action, card_pos):
        # does this card (in this position) match the given hint?
        assert action.type == Action.HINT
        matches = self.matches(action.color, action.number)
        return card_pos in action.hinted_card_pos and matches or card_pos not in action.hinted_card_pos and not matches
    
    
    def playable(self, board):
        # is this card playable on the board?
        return self.number == board[self.color] + 1
    
    def useful(self, board, full_deck, discard_pile):
        # is this card still useful?
        
        # check that lower cards still exist
        for number in xrange(board[self.color] + 1, self.number):
            copies_in_deck = sum(1 for card in full_deck if card.equals(Card(id=-1, color=self.color, number=number)))
            copies_in_discard_pile = sum(1 for card in discard_pile if card.equals(Card(id=-1, color=self.color, number=number)))
            
            if copies_in_deck == copies_in_discard_pile:
                # some lower card was discarded!
                return False
        
        return self.number > board[self.color]
    
    def relevant(self, board, full_deck, discard_pile):
        # is this card the last copy available?
        copies_in_deck = sum(1 for card in full_deck if self.equals(card))
        copies_in_discard_pile = sum(1 for card in discard_pile if self.equals(card))
        
        # this method should always be called on a card which is not discarded yet
        assert copies_in_deck > copies_in_discard_pile
        
        return self.useful(board, full_deck, discard_pile) and copies_in_deck == copies_in_discard_pile + 1



def deck():
    deck = []
    id = 0
    for color in Card.COLORS:
        for number in xrange(1, 6):
            if color == Card.RAINBOW:
                quantity = 1
            elif number == 1:
                quantity = 3
            elif 2 <= number <= 4:
                quantity = 2
            elif number == 5:
                quantity = 1
            else:
                raise Exception("Unknown card parameters.")
            
            for i in xrange(quantity):
                deck.append(Card(id, color, number))
                id += 1
    
    return deck

