#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random

from ...action import Action, PlayAction, DiscardAction, HintAction
from ...card import Card, deck
from ...base_strategy import BaseStrategy



class Knowledge:
    """
    An instance of this class represents what a player knows about his hand.
    """
    
    PERSONAL = 'personal'
    PUBLIC = 'public'
    
    TYPES = [PERSONAL, PUBLIC]
    
    def __init__(self, strategy, type, player_id, hand):
        self.strategy = strategy
        assert type in self.TYPES
        self.type = type
        self.player_id = player_id
        self.hand = hand
        
        # possibilities for each card
        self.possibilities = [set(self.strategy.full_deck) for j in xrange(self.strategy.k)]
    
    
    def __repr__(self):
        return "Knowledge of player %d (%s)" % (self.player_id, self.type)
    
    
    def reset(self, pos):
        """
        Reset possibilities for the card in the given position.
        """
        if self.hand[pos] is None:
            self.possibilites[pos] = set()
        else:
            self.possibilities[pos] = set(self.strategy.full_deck)
    
    
    def update(self):
        if self.type == self.PERSONAL:
            visible_cards = list(self.strategy.visible_cards())
        else:
            visible_cards = self.strategy.discard_pile
        
        for (j, p) in enumerate(self.possibilities):
            self.update_possibilities(p, visible_cards)
            assert len(p) > 0 or self.hand[j] is None
    
    
    def update_with_hint(self, action):
        """
        Update using hint.
        """
        for (i, p) in enumerate(self.possibilities):
            for card in self.strategy.full_deck:
                if card in p and not card.matches_hint(action, i):
                    # if self.strategy.id == 0:
                    #     self.strategy.log("Removing card %r from position %d" % (card, i))
                    p.remove(card)
    
    
    def update_possibilities(self, p, visible_cards):
        """
        Update possibilities for a single card removing visible cards.
        """
        for card in visible_cards:
            if card in p:
                p.remove(card)
    
    
    def get_unique_possibilities(self, p):
        """
        Return a list of unique cards.
        """
        res = []
        for card in p:
            if not any(c.equals(card) for c in res):
                res.append(card)
        return res
    
    def log(self):
        SIZE = 8
        self.strategy.log("%r" % self)
        for (i, p) in enumerate(self.possibilities):
            unique_p = self.get_unique_possibilities(p)
            self.strategy.log("[Card %d] " % i + ", ".join("%r" % card for card in unique_p[:SIZE]) + (", ... (%d possibilities)" % len(unique_p) if len(unique_p) > SIZE else ""))
        self.strategy.log("")


class Strategy(BaseStrategy):
    """
    An instance of this class represents a player's strategy.
    It only has the knowledge of that player, and it must make decisions.
    """
    
    def initialize(self, id, num_players, k, hands, board, discard_pile):
        """
        To be called once before the beginning.
        """
        self.id = id
        self.num_players = num_players
        self.k = k  # number of cards per hand
        self.my_hand = [None] * k   # says in which positions there is actually a card
        self.hands = hands  # hands of other players
        self.board = board
        self.discard_pile = discard_pile
        
        # store a copy of the full deck
        self.full_deck = deck()
        
        # personal knowledge
        self.personal_knowledge = Knowledge(self, Knowledge.PERSONAL, id, self.my_hand)
        
        # public knowledge
        self.public_knowledge = [Knowledge(self, Knowledge.PUBLIC, i, hands[i] if i != id else self.my_hand) for i in xrange(num_players)]
        
        # all knowledge
        self.all_knowledge = self.public_knowledge + [self.personal_knowledge]
        
        # remove cards of other players from possibilities
        self.update_knowledge()
    
    
    def update_knowledge(self):
        """
        Update all knowledge.
        """
        for kn in self.all_knowledge:
            kn.update()
    

    def visible_cards(self):
        """
        Generator of all the cards visible by me.
        """
        for card in self.discard_pile:
            yield card
        
        for hand in self.hands.itervalues():
            for card in hand:
                yield card
    
    
    
    def feed_turn(self, player_id, action):
        """
        Receive information about a played turn.
        """
        
        if action.type in [Action.PLAY, Action.DISCARD]:
            # reset knowledge of the player
            self.public_knowledge[player_id].reset(action.card_pos)
            
            if player_id == self.id:
                # check for my new card
                self.personal_knowledge.reset(action.card_pos)
        
        
        elif action.type == Action.HINT:
            # someone gave a hint!
            for kn in self.all_knowledge:
                if kn.player_id == action.player_id:
                    kn.update_with_hint(action)
        
        # update knowledge
        self.update_knowledge()
        
        if self.id == 0:
            for kn in self.all_knowledge:
                kn.log()
    
    
    def get_turn_action(self):
        """
        Decide action.
        """
        
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
            
            self.log("give some random hint")
            return HintAction(player_id, color=color, number=number)
        
        elif random.randint(0,1) == 0:
            # play random card
            card_pos = random.choice([c_pos for (c_pos, value) in enumerate(self.my_hand) if value is not None])
            self.log("play some random card")
            return PlayAction(card_pos)
        
        else:
            # discard random card
            card_pos = random.choice([c_pos for (c_pos, value) in enumerate(self.my_hand) if value is not None])
            self.log("discard some random card")
            return DiscardAction(card_pos)



