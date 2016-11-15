#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from game.card import *

my_deck = []
model_deck = set(deck())

def get_card():
    # read and validate new card from stdin
    global my_deck
    
    while True:
        line = sys.stdin.readline().split()
        try:
            number, color = line
            number = int(number)
            assert 1 <= number <= Card.NUM_NUMBERS
            assert color in Card.COLORS
        
        except Exception:
            print "Unrecognized card, try again (format: 1/2/3/4/5 %s)" % "/".join(Card.COLORS)
            continue
    
        print "Recognized card: %d %s" % (number, color)
        print "Press ENTER to confirm, or anything else (and ENTER) to undo"
        
        line = sys.stdin.readline().strip()
        if len(line) > 0:
            print "Last entered card (%d %s) is ignored, please enter the right card" % (number, color)
            continue
        
        else:
            break
    
    print "Card %d %s entered successfully" % (number, color)
    my_deck.append((number, color))


def space():
    # print white lines
    for _ in xrange(100):
        print
    



filename = sys.argv[1]
print "The new deck will be saved to", filename

with open(filename, "w") as file:

    for i in xrange(5):
        print
        print "Insert cards for player %d" % i
        print
        
        for j in xrange(4):
            get_card()
        
        space()

    while len(my_deck) < len(model_deck):
        print
        print "Insert new card"
        print
        get_card()
        space()


    print "Hopefully this was the last card of the deck!"
    print "Press ENTER to continue, after you finished the game"

    sys.stdin.readline()

    print
    for number, color in my_deck:
        print number, color
    
    for number, color in reversed(my_deck):
        card = None
        for c in model_deck:
            if c.number == number and c.color == color:
                card = c
                break
        
        if card is None:
            print "ERROR: inconsistent deck (too many %d %s)" % (number, color)
            sys.exit(1)
        
        model_deck.remove(card)
        print >> file, "%d %s %d" % (card.number, card.color, card.id)
        
    assert len(model_deck) == 0
    print "Deck successfully saved to", filename


