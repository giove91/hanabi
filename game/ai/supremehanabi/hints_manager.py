#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import copy
from collections import Counter, OrderedDict

from ...action import Action, PlayAction, DiscardAction, HintAction
from ...card import Card
from ...game import Game

from constants import *


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
        assert 0 <= secondary < self.cards_per_player
    
    
    def __add__(self, other):
        return HintInformation(
            k = self.k,
            primary = (self.primary + other.primary) % (2 * (self.k - 1)),
            secondary = (self.secondary + other.secondary) % self.cards_per_player
        )
    
    def __radd__(self, other):
        if other is 0:
            return self
        else:
            return self.__add__(other)
    
    def __sub__(self, other):
        return HintInformation(
            k = self.k,
            primary = (self.primary - other.primary) % (2 * (self.k - 1)),
            secondary = (self.secondary - other.secondary) % self.cards_per_player
        )
    
    def __hash__(self):
        return self.primary * self.cards_per_player + self.secondary
    
    def __eq__(self, other):
        return self.primary, self.secondary == other.primary, other.secondary   # k is fixed during a game
    

class Meaning(dict):
    """
    Dictionary of the form
        {card_pos: set of possible cards in this position}.
    Supports + (it is actually an "or") operation.
    """
    
    def __add__(self, other):
        return Meaning(
            (card_pos, cards | other[card_pos]) for (card_pos, cards) in self.iteritems() if card_pos in other
        )
    
    def __radd__(self, other):
        if other is 0:
            return self
        else:
            return self.__add__(other)


def hint_informations(k):
    """
    Generate all possible informations.
    """
    cards_per_player = Game.CARDS_PER_PLAYER[k]
    
    for primary in xrange(2*(k-1)):
        for secondary in xrange(cards_per_player):
            yield HintInformation(k, primary, secondary)



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
        
        # compute number of primary and secondary slots
        self.num_primary = 2*(self.k-1)
        self.num_secondary = Game.CARDS_PER_PLAYER[self.k]
    
    
    def log(self, message):
        self.strategy.log(message)
    
    
    def compute_information_meanings(self, player_id):
        """
        For the given player, decide what every possible information means.
        Can use:
        - all public knowledge;
        - status of the game (turn number, deck size, discard pile, board, number of hints).
        
        Return something like the following (every information restricts some cards to some values):
        {
            Information(0,0): Meaning({
                Card 2: set([2 Yellow, 3 Red, 1 White]),
                Card 3: set([5 Rainbow])
            }),
            Information(0,1): Meaning({
                ...
            })
            ...
        
        At least one case should apply.
        """
        
        knowledge = self.public_knowledge[player_id]
        overview = knowledge.overview()
        
        # assign ranks to each position
        # higher rank makes it more likely that we want to give information about that card
        ranks = [0 for card_pos in xrange(self.k)]
        
        ### ranks are as follows ###
        for card_pos in xrange(self.k):
            o = overview[card_pos]
            
            if knowledge.hand()[card_pos] is None:
                # 0: no card in the given position
                rank = 0
            
            elif sum(o.itervalues()) == 1:
                # 1: card is known exactly
                rank = 1
            
            elif o[PLAYABLE] + o[RELEVANT] + o[USEFUL] == 0:
                # 2: card is known to be useless
                rank = 2
            
            elif o[PLAYABLE] == 0:
                # 3: card is known to be non-playable
                rank = 3
            
            elif o[USELESS] + o[RELEVANT] + o[USEFUL] == 0:
                # 4: card is known to be playable
                rank = 4
            
            else:
                # 5: none of the previous cases apply
                rank = 5
            
            ranks[card_pos] = rank
        
        # give hint on the highest ranked position
        card_pos = ranks.index(max(ranks))
        
        meanings = OrderedDict((information, Meaning()) for information in hint_informations(self.k))
        
        if sum(o.itervalues()) <= self.num_primary:
            # give one possibility to each primary slot
            
            p_list = knowledge.possibilities[card_pos].keys()
            
            for primary in xrange(self.num_primary):
                cards = set([p_list[primary]]) if primary < len(p_list) else set()
                for secondary in xrange(self.num_secondary):
                    meanings[HintInformation(self.k, primary, secondary)][card_pos] = cards
        
        else:
            # TODO
            pass
        
        return meanings
    
    
    def compute_information(self, player_id, meanings=None):
        """
        Compute information for the given player (different from myself).
        If given, use the already computed meanings.
        """
        assert player_id != self.id
        hand = self.strategy.hands[player_id]
        
        if meanings is None:
            meanings = self.compute_information_meanings(player_id)
        
        for (information, meaning) in meanings.iteritems():
            # does this meaning apply?
            if all(hand[card_pos] is None or hand[card_pos] in cards for card_pos, cards in meaning.iteritems()):
                # yes
                return information
        
        # strange, no information applied?
        # close the baracca
        assert False


    def compute_hint(self):
        """
        Compute hint to give to some other player.
        """
        other_players_id = self.strategy.other_players_id()
        s = sum(self.compute_information(player_id) for player_id in other_players_id)
        
        # choose hint type
        hint_type = Action.HINT_TYPES[s.primary % 2]
        
        # choose player
        player_id = other_players_id[s.primary / 2]
        
        # choose value
        value = self.strategy.hands[player_id][s.secondary].value(hint_type)
        
        return HintAction(player_id, hint_type=hint_type, value=value)
    
    
    def process_hint(self, hinter_id, hint_action):
        """
        Process hint given by someone.
        """
        hinted_players_id = self.strategy.other_players_id(exclude=hinter_id)
        
        # transform hint_action into informations
        primary = Action.HINT_TYPES.index(hint_action.hint_type) + 2 * hinted_players_id.index(hint_action.player_id)
        secondaries = hint_action.cards_pos
        informations_sum = [HintInformation(self.k, primary, secondary) for secondary in secondaries]
        
        # compute meanings for every hinted player
        players_to_meanings = {
            player_id: self.compute_information_meanings(player_id) for player_id in hinted_players_id
        }
        
        # compute information of all the other players (from their hands)
        visible_informations = {
            player_id: self.compute_information(player_id, players_to_meanings[player_id]) \
            for player_id in self.strategy.other_players_id() if player_id != hinter_id
        }
        
        
        # deduce things on my hand and update personal knowledge (if I am not the hinter)
        if hinter_id != self.id:
            my_informations = [information_sum - sum(visible_informations.itervalues()) for information_sum in informations_sum]    # the possible informations about my hand
            
            my_meaning = sum(players_to_meanings[self.id][information] for information in my_informations)
            
            # update personal knowledge
            self.strategy.personal_knowledge.update_with_meaning(my_meaning)
        
        
        # update public knowledge
        for player_id in hinted_players_id:
            if len(informations_sum) == 1:
                # both primary and secondary information are public
                meaning = visible_informations[player_id] if player_id != self.id else my_meaning
            
            else:
                # only primary information is public!
                primary = visible_informations[player_id].primary if player_id != self.id else my_informations[0].primary
                meaning = sum(players_to_meanings[player_id][HintInformation(self.k, primary, secondary)] for secondary in xrange(self.num_secondary))
                
                # update public knowledge
                self.strategy.public_knowledge[player_id].update_with_meaning(meaning)



