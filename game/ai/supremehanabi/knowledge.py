#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from operator import mul
import itertools

from constants import *


class Knowledge:
    """
    An instance of this class represents what a player knows about his hand.
    """
    
    PERSONAL = 'personal'
    PUBLIC = 'public'
    
    TYPES = [PERSONAL, PUBLIC]
    
    # maximum number of combinations that can be considered
    MAX_COMBINATIONS = 100
    
    def __init__(self, strategy, type, player_id):
        self.strategy = strategy
        assert type in self.TYPES
        self.type = type
        self.player_id = player_id
        
        # possibilities for each card
        self.possibilities = [Counter(self.strategy.full_deck) for j in xrange(self.strategy.k)]
        
        # combinations (initially not constructed)
        self.combinations = None
    
    
    def __repr__(self):
        return "Knowledge of player %d (%s)" % (self.player_id, self.type)
    
    
    def hand(self):
        if self.player_id == self.strategy.id:
            return self.strategy.my_hand
        else:
            return self.strategy.hands[self.player_id]
    
    
    def reset(self, pos):
        """
        Reset possibilities for the card in the given position.
        This method is called every time someone plays or discard a card.
        """
        if self.hand()[pos] is None:
            self.possibilities[pos] = Counter()
        else:
            self.possibilities[pos] = Counter(self.strategy.full_deck)
    
    
    def get_possible_cards(self):
        """
        Get Counter of possible cards, based on visible cards.
        """
        visible_cards = None
        possible_cards = None
        
        if self.type == self.PERSONAL:
            visible_cards = self.strategy.visible_cards()
        
        else:
            if self.strategy.deck_size > 0:
                visible_cards = Counter(self.strategy.discard_pile)
            
            else:
                # in the last round, cards of the players can be considered as publicly visible
                if self.player_id == self.strategy.id:
                    # my knowledge
                    visible_cards = self.strategy.visible_cards()
                
                else:
                    # knoledge of some other player
                    possible_cards = Counter(self.strategy.hands[self.player_id])
                    del possible_cards[None]
        
        if possible_cards is None:
            possible_cards = self.strategy.full_deck_composition - visible_cards
        
        return possible_cards
    
    
    def update_with_visible_cards(self):
        """
        Update using visible cards.
        This method is called every time someone plays or discard a card.
        """
        possible_cards = self.get_possible_cards()
        
        for card_pos in xrange(self.strategy.k):
            self.update_single_card(card_pos, possible_cards)
            assert len(self.possibilities[card_pos]) > 0 or self.hand()[card_pos] is None
    
    
    def update_with_hint(self, action):
        """
        Update using hint.
        This method is called every time someone gives a hint.
        """
        for (card_pos, p) in enumerate(self.possibilities):
            self.possibilities[card_pos] = Counter(
                {card: v for (card, v) in p.iteritems() if card.matches_hint(action, card_pos)}
            )
    
    
    def update_single_card(self, card_pos, possible_cards):
        """
        Update possibilities for a single card, restricting to the Counter of the given possible cards.
        """
        self.possibilities[card_pos] &= possible_cards
    
    
    def update(self):
        """
        This method is called once at the end of feed_turn, to normalize knowledge.
        """
        # possibly update combinations
        self.update_combinations()
    
    
    def update_combinations(self):
        """
        Create/update/delete storage of all possible combinations of the hand.
        This method is called by update().
        """
        hand = self.hand()
        num_combinations = reduce(mul, [len(p) for p in self.possibilities if len(p) > 0], 1)
        
        if num_combinations <= self.MAX_COMBINATIONS:
            # look for valid combinations
            # self.combinations = []
            positions = [card_pos for card_pos in xrange(self.strategy.k) if hand[card_pos] is not None]
            possible_cards = self.get_possible_cards()
            possible_cards_per_position = [set() for i in xrange(len(positions))]
        
            for combination in itertools.product(*[p for p in self.possibilities if len(p) > 0]):
                # check that this combination is valid
                if all(possible_cards[card] >= v for (card, v) in Counter(combination).iteritems()):
                    for (i, card) in enumerate(combination):
                        possible_cards_per_position[i].add(card)
                    # self.combinations.append(combination)
            
            # update possibilities
            for (i, card_pos) in enumerate(positions):
                # possible_cards_here_set = set(combination[i] for combination in self.combinations)
                # possible_cards_here = Counter({card: possible_cards[card] for card in possible_cards_here_set})
                possible_cards_here = Counter({card: possible_cards[card] for card in possible_cards_per_position[i]})
                
                # print possible_cards_here
                self.update_single_card(card_pos, possible_cards_here)
        
        else:
            # delete stored data as it is outdated
            self.combinations = None
            self.weights = None
    
    
    def log(self, verbose=0):
        if verbose >= 1:
            SIZE = 8
            self.strategy.log("%r" % self)
            for (i, p) in enumerate(self.possibilities):
                self.strategy.log("[Card %d] " % i + ", ".join("%r" % card for card in sorted(p.keys())[:SIZE]) + (", ... (%d possibilities)" % len(p) if len(p) > SIZE else ""))
            self.strategy.log("")
        
        else:
            self.strategy.log("%r: %r" % (self, [len(p) for p in self.possibilities]))
    
    
    def overview(self):
        """
        For each card position, return the number of possible:
        - useless cards;
        - playable cards;
        - relevant not playable cards;
        - useful not playable not relevant cards.
        """
        res = [{
                USELESS: 0,
                PLAYABLE: 0,
                RELEVANT: 0,
                USEFUL: 0
            } for i in xrange(self.strategy.k)]
        
        board = self.strategy.board
        full_deck = self.strategy.full_deck_composition
        discard_pile = self.strategy.discard_pile_composition
        
        for card_pos, p in enumerate(self.possibilities):
            for card in p.iterkeys():
                if not card.useful(board, full_deck, discard_pile):
                    res[card_pos][USELESS] += 1
                elif card.playable(board):
                    res[card_pos][PLAYABLE] += 1
                elif card.relevant(board, full_deck, discard_pile):
                    res[card_pos][RELEVANT] += 1
                else:
                    res[card_pos][USEFUL] += 1
        
        return res
    
    
    def playable_probability(self, card_pos):
        """
        Probability that a card is playable.
        """
        if self.hand()[card_pos] is None:
            return None
        
        p = self.possibilities[card_pos]
        return float(sum(v for (card, v) in p.iteritems() if card.playable(self.strategy.board))) / sum(p.values())
    
    
    def playable(self, card_pos):
        """
        Is this card surely playable?
        """
        if self.hand()[card_pos] is None:
            return None
        
        return all(card.playable(self.strategy.board) for card in self.possibilities[card_pos])


