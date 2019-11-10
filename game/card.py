#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from collections import Counter
from functools import total_ordering

from action import Action


@total_ordering
class CardAppearance:
    """
    A card as seen by a player.
    It only has color and number, but not the id.
    """
    RED = 'Red'
    BLUE = 'Blue'
    WHITE = 'White'
    YELLOW = 'Yellow'
    GREEN = 'Green'
    RAINBOW = 'Rainbow'
    
    NUM_COLORS = 6
    NUM_NUMBERS = 5
    
    COLORS = [RED, BLUE, WHITE, YELLOW, GREEN, RAINBOW]
    
    COLORS_TO_NUMBERS = {color: i for (i, color) in enumerate(COLORS)}
    
    PRINTABLE_COLORS = {
        RED: 'red',
        BLUE: 'blue',
        WHITE: 'grey',
        YELLOW: 'yellow',
        GREEN: 'green',
        RAINBOW: 'magenta'
    }
    
    
    def __init__(self, color, number):
        assert color in self.COLORS
        assert 1 <= number <= self.NUM_NUMBERS
        
        self.color = color
        self.number = number
    
    
    def __repr__(self):
        from termcolor import colored
        return colored("%d %s" % (self.number, self.color), self.PRINTABLE_COLORS[self.color])
    
    def __hash__(self):
        return self.COLORS_TO_NUMBERS[self.color] + self.NUM_COLORS * self.number
    
    def __eq__(self, other):
        # same color and number
        return isinstance(other, CardAppearance) and self.color == other.color and self.number == other.number

    def __ne__(self, other):
        return self != other
    
    def __le__(self, other):
        if other is None:
            return False
        return (self.color, self.number) < (other.color, other.number)
    
    
    def equals(self, other):
        # same color and number (but possibly different id)
        # (this method is inherited by Card, but it exists here
        # so that it can be used both by CardAppearance and Card)
        return self.color == other.color and self.number == other.number
    
    
    def matches(self, color=None, number=None):
        # does this card match the given color/number? (if the value is None, it is considered to match)
        if color is not None and self.color != color:
            return False
        if number is not None and self.number != number:
            return False
        return True
    
    
    def matches_hint(self, action, card_pos):
        # does this card (in the given position) match the given hint?
        assert action.type == Action.HINT
        matches = self.matches(action.color, action.number)
        return card_pos in action.cards_pos and matches or card_pos not in action.cards_pos and not matches
    
    
    def value(self, type):
        if type == Action.COLOR:
            return self.color
        else:
            return self.number
    
    
    def appearance(self):
        """
        This method is overloaded by Card.
        """
        return self
    
    """
    The following methods are only used by AIs, not by Game.
    """
    
    def playable(self, board):
        """
        Is this card playable on the board?
        """
        return self.number == board[self.color] + 1
    
    
    def useful(self, board, full_deck, discard_pile):
        """
        Is this card still useful?
        full_deck and discard_pile can be given either as lists or as Counters (more efficient).
        """
        # check that lower cards still exist
        for number in xrange(board[self.color] + 1, self.number):
            if isinstance(full_deck, Counter):
                copies_in_deck = full_deck[CardAppearance(color=self.color, number=number)]
            else:
                copies_in_deck = sum(1 for card in full_deck if card.equals(CardAppearance(color=self.color, number=number)))
            
            if isinstance(discard_pile, Counter):
                copies_in_discard_pile = discard_pile[CardAppearance(color=self.color, number=number)]
            else:
                copies_in_discard_pile = sum(1 for card in discard_pile if card.equals(CardAppearance(color=self.color, number=number)))
            
            if copies_in_deck == copies_in_discard_pile:
                # some lower card was discarded!
                return False
        
        return self.number > board[self.color]
    
    
    def relevant(self, board, full_deck, discard_pile):
        """
        Is this card the last copy available?
        full_deck and discard_pile can be given either as lists or as Counters (more efficient).
        """
        if isinstance(full_deck, Counter):
            copies_in_deck = full_deck[self.appearance()]
        else:
            copies_in_deck = sum(1 for card in full_deck if self.equals(card))
        
        if isinstance(discard_pile, Counter):
            copies_in_discard_pile = discard_pile[self.appearance()]
        else:
            copies_in_discard_pile = sum(1 for card in discard_pile if self.equals(card))
        
        return self.useful(board, full_deck, discard_pile) and copies_in_deck == copies_in_discard_pile + 1



class Card(CardAppearance):
    """
    A real card, with id.
    """
    
    def __init__(self, id, color, number):
        assert color in self.COLORS
        assert 1 <= number <= self.NUM_NUMBERS
        
        self.id = id
        self.color = color
        self.number = number
    
    def __hash__(self):
        return self.id
    
    def __eq__(self, other):
        # same id
        return self.id == other.id

    def __ne__(self, other):
        return self != other
    
    
    def appearance(self):
        """
        Construct the corresponding CardAppearance (forget id).
        """
        return CardAppearance(self.color, self.number)



def get_appearance(cards, hide=False):
    """
    Given a list of (possibly None) Card objects, return the list of CardAppearance objects.
    If hide=True, then hide the card and put 0 instead (this is used by Player to hide cards).
    """
    if hide:
        return [0 if card is not None else None for card in cards]
    else:
        return [card.appearance() if card is not None else None for card in cards]



