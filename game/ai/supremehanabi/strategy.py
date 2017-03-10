#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random
from collections import Counter
from operator import mul
import itertools

from ...action import Action, PlayAction, DiscardAction, HintAction
from ...card import Card, get_appearance
from ...deck import DECKS
from ...base_strategy import BaseStrategy

from .knowledge import Knowledge
from .hints_manager import HintsManager


class Strategy(BaseStrategy):
    """
    An instance of this class represents a player's strategy.
    It only has the knowledge of that player, and it must make decisions.
    """
    TOLERANCE = 0.001
    
    def initialize(self, id, num_players, k, board, deck_type, my_hand, hands, discard_pile, deck_size):
        """
        To be called once before the beginning.
        """
        self.id = id
        self.num_players = num_players
        self.k = k  # number of cards per hand
        self.board = board
        self.deck_type = deck_type
        
        # store a copy of the full deck
        self.full_deck = get_appearance(DECKS[deck_type]())
        self.full_deck_composition = Counter(self.full_deck)
        
        # hands
        self.my_hand = my_hand  # says in which positions there is actually a card
        self.hands = hands
        
        # discard pile
        self.discard_pile = discard_pile
        
        # deck size
        self.deck_size = deck_size
        
        # personal knowledge
        self.personal_knowledge = Knowledge(self, Knowledge.PERSONAL, id)
        
        # public knowledge
        self.public_knowledge = [Knowledge(self, Knowledge.PUBLIC, i) for i in xrange(num_players)]
        
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
        Counter of all the cards visible by me.
        """
        res = Counter(self.discard_pile)
        for hand in self.hands.itervalues():
            res += Counter(hand)
        
        return res
    
    
    def next_player_id(self):
        return (self.id + 1) % self.num_players
    
    def other_players_id(self):
        return [i for i in xrange(self.num_players) if i != self.id]
    
    
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
            
            # new cards are visible
            for kn in self.all_knowledge:
                kn.update_with_visible_cards()
        
        
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
        Choose action for this turn.
        """
        # check for playable cards in my hand
        for card_pos in xrange(self.k):
            if self.personal_knowledge.playable(card_pos):
                return PlayAction(card_pos=card_pos)
        
        if self.hints >= 1:
            # give hint, looking for some unknown playable card
            for player_id in self.other_players_id():
                for (card_pos, card) in enumerate(self.hands[player_id]):
                    if card is None:
                        continue
                    
                    kn = self.public_knowledge[player_id]   # public knowledge of that player
                    if card.playable(self.board) and not kn.playable(card_pos):
                        if any(c.color != card.color for c in kn.possibilities[card_pos]):
                            # hint on color
                            return HintAction(player_id=player_id, color=card.color)
                        else:
                            # hint on number
                            return HintAction(player_id=player_id, number=card.number)
        
        # discard card
        for card_pos in xrange(self.k):
            if self.my_hand[card_pos] is not None:
                return DiscardAction(card_pos=card_pos)


