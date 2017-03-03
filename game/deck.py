#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .card import Card


def standard_deck(include_rainbow=True):
    deck = []
    id = 0
    for color in Card.COLORS:
        if not include_rainbow and color == Card.RAINBOW:
            continue
        
        for number in xrange(1, Card.NUM_NUMBERS + 1):
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
    
    if include_rainbow:
        assert len(deck) == 55
    else:
        assert len(deck) == 50
    
    return deck


def standard_deck_25():
    return standard_deck(include_rainbow=False)


DECK55 = 'deck55'
DECK50 = 'deck50'

DECKS = {
    DECK55: standard_deck,
    DECK50: standard_deck_25
}

