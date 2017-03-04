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



class Knowledge:
    """
    An instance of this class represents what a player knows about his hand.
    """
    
    PERSONAL = 'personal'
    PUBLIC = 'public'
    
    TYPES = [PERSONAL, PUBLIC]
    
    # maximum number of combinations that can be considered
    MAX_COMBINATIONS = 1000
    
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
            # compute and store valid combinations
            self.combinations = []
            positions = [card_pos for card_pos in xrange(self.strategy.k) if hand[card_pos] is not None]
            possible_cards = self.get_possible_cards()
        
            for combination in itertools.product(*[p for p in self.possibilities if len(p) > 0]):
                # check that this combination is valid
                if all(possible_cards[card] >= v for (card, v) in Counter(combination).iteritems()):
                    self.combinations.append(combination)
                
            self.strategy.log("Found %d valid combinations out of %d." % (len(self.combinations), num_combinations))
            
            # update possibilities
            for (i, card_pos) in enumerate(positions):
                # print i, combination
                possible_cards_here_set = set(combination[i] for combination in self.combinations)
                possible_cards_here = Counter({card: possible_cards[card] for card in possible_cards_here_set})
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


