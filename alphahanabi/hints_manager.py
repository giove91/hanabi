#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import copy
import networkx as nx

sys.path.append("..") 


from action import Action, PlayAction, DiscardAction, HintAction
from card import Card


class BaseHintsManager(object):
    """
    Base class for a HintsManager.
    """
    def __init__(self, strategy):
        self.strategy = strategy    # my strategy object
        
        # copy something from the strategy
        self.id = strategy.id
        self.num_players = strategy.num_players
        self.k = strategy.k
        self.my_hand = strategy.my_hand
        self.hands = strategy.hands
        self.possibilities = strategy.possibilities
        self.full_deck = strategy.full_deck
        self.board = strategy.board
        self.knowledge = strategy.knowledge
        
        self.COLORS_TO_NUMBERS = {color: i for (i, color) in enumerate(Card.COLORS)}
    
    
    def log(self, message):
        self.strategy.log(message)
    
    
    def is_appropriate(self, player_id, action):
        """
        Returns True iff the given hint should be processed by this HintsManager.
        """
        raise NotImplementedError
    
    
    def receive_hint(self, player_id, action):
        """
        Receive hint given by player_id.
        """
        if action.player_id == self.id:
            # process direct hint
            for (i, p) in enumerate(self.possibilities):
                for card in self.full_deck:
                    if not card.matches_hint(action, i) and card in p:
                        # self.log("removing card %r from position %d due to hint" % (card, i))
                        p.remove(card)
        
        # update knowledge
        for card_pos in action.cards_pos:
            kn = self.knowledge[action.player_id][card_pos]
            if action.hint_type == Action.COLOR:
                kn.color = True
            else:
                kn.number = True
        
        assert self.possibilities is self.strategy.possibilities
        assert self.board is self.strategy.board
        assert self.knowledge is self.strategy.knowledge
    
    
    def get_hint(self):
        """
        Compute hint to give.
        """
        raise NotImplementedError
    




class ValueHintsManager(BaseHintsManager):
    """
    Value hints manager.
    A hint communicates to every other player the value (color or number) of one of his cards.
    
    More specifically, the players agree on a function player->card_pos (which depends on the turn and on other things).
    The current player computes the sum of the values (color or number) of the agreed cards,
    and gives a hint on that value.
    Then each of the other players deduces the value of his card.
    """
    
    def __init__(self, *args, **kwargs):
        super(ValueHintsManager, self).__init__(*args, **kwargs)
        self.COLORS_TO_NUMBERS = {color: i for (i, color) in enumerate(Card.COLORS)}
    
    
    def is_appropriate(self, player_id, action):
        """
        Returns True iff the given hint should be processed by this HintsManager.
        """
        return not (self.num_players == 5 and self.k == 4 and action.turn == 0)
    
    
    def shift(self, turn):
        # a variable shift in the hint
        return turn + turn / self.num_players
    
    
    def is_duplicate(self, card):
        """
        Says if the given card is owned by some player who knows everything about it.
        """
        # check other players
        for (player_id, hand) in self.strategy.hands.iteritems():
            for card_pos in xrange(self.k):
                kn = self.knowledge[player_id][card_pos]
                if kn.color and kn.number and hand[card_pos] is not None and hand[card_pos].equals(card):
                    return True
        
        # check my hand
        for card_pos in xrange(self.k):
            kn = self.knowledge[self.id][card_pos]
            if kn.color and kn.number and any(card.equals(c) for c in self.strategy.possibilities[card_pos]):
                return True
        
        return False
    
    
    def choose_card(self, player_id, target_id, turn, hint_type):
        """
        Choose which of the target's cards receive a hint from the current player in the given turn.
        """
        hand = new_card = self.my_hand if target_id == self.id else self.hands[target_id]
        possible_cards = [card_pos for (card_pos, kn) in enumerate(self.knowledge[target_id]) if hand[card_pos] is not None and not (kn.color if hint_type == Action.COLOR else kn.number)]
        
        if len(possible_cards) == 0:
            # do not give hints
            return None
        
        # TODO: forse usare un vero hash
        n = turn * 11**3 + (0 if hint_type == Action.COLOR else 1) * 119 + player_id * 11 + target_id
        
        return possible_cards[n % len(possible_cards)]
    
    
    def choose_all_cards(self, player_id, turn, hint_type):
        """
        Choose all cards that receive hints (of the given type) from the given player in the given turn.
        """
        return {target_id: self.choose_card(player_id, target_id, turn, hint_type) for target_id in xrange(self.num_players) if target_id != player_id and self.choose_card(player_id, target_id, turn, hint_type) is not None}
    
    
    def infer_playable_cards(self, player_id, action):
        """
        From the choice made by the hinter (give hint on color or number), infer something
        about the playability of my cards.
        Here it is important that:
        - playability of a card depends only on things that everyone sees;
        - the choice of the type of hint (color/number) is primarily based on the number of playable cards.
        Call this function before decode_hint(), i.e. before knowledge is updated.
        """
        hint_type = action.hint_type
        opposite_hint_type = Action.NUMBER if hint_type == Action.COLOR else Action.COLOR
        
        cards_pos = self.choose_all_cards(player_id, action.turn, hint_type)
        alternative_cards_pos = self.choose_all_cards(player_id, action.turn, opposite_hint_type)
        
        if self.id not in cards_pos or self.id not in alternative_cards_pos:
            # I already knew about one of the two cards
            return None

        if action.player_id == self.id:
            # the hint was given to me, so I haven't enough information to infer something
            return None
        
        if hint_type == Action.NUMBER:
            # the alternative hint would have been on colors
            visible_colors = set(card.color for (i, hand) in self.strategy.hands.iteritems() for card in hand if i != player_id and card is not None)   # numbers visible by me and by the hinter
            if len(visible_colors) < Card.NUM_COLORS:
                # maybe the hinter was forced to make his choice because the color he wanted was not available
                return None
            
        else:
        # the alternative hint would have been on numbers
            visible_numbers = set(card.number for (i, hand) in self.strategy.hands.iteritems() for card in hand if i != player_id and card is not None)   # numbers visible by me and by the hinter
            if len(visible_numbers) < Card.NUM_NUMBERS:
                # maybe the hinter was forced to make his choice because the number he wanted was not available
                return None
        
        
        involved_cards = [hand[cards_pos[i]] for (i, hand) in self.strategy.hands.iteritems() if i != player_id and i in cards_pos] + [self.strategy.hands[action.player_id][card_pos] for card_pos in action.cards_pos]
        
        involved_cards = list(set(involved_cards))
        my_card_pos = cards_pos[self.id]
        num_playable = sum(1 for card in involved_cards if card.playable(self.strategy.board) and not self.is_duplicate(card))
        
        alternative_involved_cards = [hand[alternative_cards_pos[i]] for (i, hand) in self.strategy.hands.iteritems() if i != player_id and i in alternative_cards_pos]
        alternative_my_card_pos = alternative_cards_pos[self.id]
        alternative_num_playable = sum(1 for card in alternative_involved_cards if card.playable(self.strategy.board) and not self.is_duplicate(card))
        
        # self.log("Num playable: %d, %d" % (num_playable, alternative_num_playable))
        # self.log("%r %r" % (involved_cards, my_card_pos))
        # self.log("%r %r" % (alternative_involved_cards, alternative_my_card_pos))
        
        if alternative_num_playable > num_playable:
            assert alternative_num_playable == num_playable + 1
            # found a playable card and a non-playable card!
            self.log("found playable card (%d) and non-playable card (%d)" % (my_card_pos, alternative_my_card_pos))
            return my_card_pos, alternative_my_card_pos
        
        

    def decode_hint(self, player_id, action):
        """
        Decode hint given by someone else (not necessarily directly to me).
        """
        hint_type = action.hint_type
        cards_pos = self.choose_all_cards(player_id, action.turn, hint_type)
        # self.log("%r" % cards_pos)
        
        # update knowledge
        for (target_id, card_pos) in cards_pos.iteritems():
            kn = self.knowledge[target_id][card_pos]
            if hint_type == Action.COLOR:
                kn.color = True
            else:
                kn.number = True
        
        # decode my hint
        if self.id in cards_pos:
            n = action.number if hint_type == Action.NUMBER else self.COLORS_TO_NUMBERS[action.color]
            my_card_pos = cards_pos[self.id]
            modulo = Card.NUM_NUMBERS if hint_type == Action.NUMBER else Card.NUM_COLORS
            
            involved_cards = [hand[cards_pos[i]] for (i, hand) in self.strategy.hands.iteritems() if i != player_id and i in cards_pos]
            
            m = sum(card.number if hint_type == Action.NUMBER else self.COLORS_TO_NUMBERS[card.color] for card in involved_cards) + self.shift(action.turn)
            my_value = (n - m) % modulo
            
            # self.log("involved_cards: %r" % involved_cards)
            # self.log("m: %d, my value: %d, shift: %d" % (m, my_value,self.shift(action.turn)))
            
            number = my_value if hint_type == Action.NUMBER else None
            if number == 0:
                number = 5
            color = Card.COLORS[my_value] if hint_type == Action.COLOR else None
            
            return my_card_pos, color, number
        
        else:
            # no hint (apparently I already know everything)
            return None
    
    
    
    def receive_hint(self, player_id, action):
        # maybe I wasn't given a hint because I didn't have the right cards
        # recall: the hint is given to the first suitable person after the one who gives the hint
        for i in range(player_id + 1, self.num_players) + range(player_id):
            if i == action.player_id:
                # reached hinted player
                break
            
            elif i == self.id:
                # I was reached first!
                # I am between the hinter and the hinted player!
                for (i, p) in enumerate(self.possibilities):
                    for card in self.full_deck:
                        if not card.matches_hint(action, -1) and card in p:
                            # self.log("removing card %r from position %d due to hint skip" % (card, i))
                            p.remove(card)
        
        # infer playability of some cards, from the type of the given hint
        res = self.infer_playable_cards(player_id, action)
        
        if res is not None:
            # found a playable and a non-playable card
            playable, non_playable = res
            for card in self.full_deck:
                if card.playable(self.board) and card in self.possibilities[non_playable] and not self.is_duplicate(card):
                    # self.log("removing %r from position %d" % (card, non_playable))
                    self.possibilities[non_playable].remove(card)
                elif not card.playable(self.board) and card in self.possibilities[playable] and not self.is_duplicate(card):
                    # self.log("removing %r from position %d" % (card, playable))
                    self.possibilities[playable].remove(card)
        
        # process value hint
        res = self.decode_hint(player_id, action)
        
        if res is not None:
            card_pos, color, number = res
            # self.log("thanks to indirect hint, understood that card %d has " % card_pos + ("number %d" % number if action.hint_type == Action.NUMBER else "color %s" % color))
        
            p = self.possibilities[card_pos]
            for card in self.full_deck:
                if not card.matches(color=color, number=number) and card in p:
                    p.remove(card)
        
        # important: this is done at the end because it changes the knowledge
        super(ValueHintsManager, self).receive_hint(player_id, action)
    
    
    def compute_hint_value(self, turn, hint_type):
        """
        Returns the color/number we need to give a hint about.
        """
        cards_pos = self.choose_all_cards(self.id, turn, hint_type)
        # self.log("cards_pos: %r" % cards_pos)
        
        if len(cards_pos) == 0:
            # the other players already know everything
            return None
        
        # compute sum of visible cards in the given positions
        modulo = Card.NUM_NUMBERS if hint_type == Action.NUMBER else Card.NUM_COLORS
        involved_cards = [hand[cards_pos[i]] for (i, hand) in self.strategy.hands.iteritems() if i in cards_pos]
        assert all(card is not None for card in involved_cards)
        m = sum(card.number if hint_type == Action.NUMBER else self.COLORS_TO_NUMBERS[card.color] for card in involved_cards) + self.shift(turn)
        m %= modulo
        
        number = m if hint_type == Action.NUMBER else None
        if number == 0:
            number = 5
        color = Card.COLORS[m] if hint_type == Action.COLOR else None
        
        return color, number
    
    
    def get_hint(self):
        """
        Choose the best hint to give, if any.
        """
        # try the two possible hint_type values
        possibilities = {hint_type: None for hint_type in Action.HINT_TYPES}
        
        for hint_type in Action.HINT_TYPES:
            # compute which cards would be involved in this indirect hint
            cards_pos = self.choose_all_cards(self.id, self.strategy.turn, hint_type)
            involved_cards = [self.strategy.hands[i][card_pos] for (i, card_pos) in cards_pos.iteritems()]
            
            res = self.compute_hint_value(self.strategy.turn, hint_type)
            
            # self.log("involved cards: %r" % involved_cards)
            # self.log("%r, shift: %d" % (res, self.shift(self.strategy.turn)))
            if res is not None:
                color, number = res
            
                # search for the first player with cards matching the hint
                player_id = None
                num_matches = None
                for i in range(self.id + 1, self.num_players) + range(self.id):
                    hand = self.strategy.hands[i]
                    num_matches = 0
                    for card in hand:
                        if card is not None and card.matches(color=color, number=number):
                            # found!
                            num_matches += 1
                    if num_matches > 0:
                        player_id = i
                        break
                
                
                if player_id is not None:
                    # found player to give the hint to
                    involved_cards += [card for (card_pos, card) in enumerate(self.strategy.hands[player_id]) if card is not None and card.matches(color=color, number=number) and not self.knowledge[player_id][card_pos].knows(hint_type)]
                    involved_cards = list(set(involved_cards))
                    
                    num_relevant = sum(1 for card in involved_cards if card.relevant(self.strategy.board, self.strategy.full_deck, self.strategy.discard_pile) and not self.is_duplicate(card))
                    num_playable = sum(1 for card in involved_cards if card.playable(self.strategy.board) and not self.is_duplicate(card))
                    num_useful = sum(1 for card in involved_cards if card.useful(self.strategy.board, self.strategy.full_deck, self.strategy.discard_pile) and not self.is_duplicate(card))
                    
                    # self.log("involved cards: %r" % involved_cards)
                    # self.log("there are %d playable, %d relevant, %d useful cards" % (num_playable, num_relevant, num_useful))
                    
                    # Give priority to playable cards, then to relevant cards, then to the number of cards.
                    # WARNING: it is important that the first parameter is the number of playable cards,
                    # because other players obtain information from this.
                    # If the hint doesn't involve any useful card, avoid giving the hint.
                    if num_useful > 0:
                        possibilities[hint_type] = (
                                (num_playable, num_relevant, len(involved_cards)),
                                HintAction(player_id=player_id, color=color, number=number)
                            )
        
        # choose between color and number
        possibilities = {a: b for (a,b) in possibilities.iteritems() if b is not None}
        

        if len(possibilities) > 0:
            score, action = sorted(possibilities.itervalues(), key = lambda x: x[0])[-1]
            self.log("giving indirect hint on %d cards with score %d, %d" % (score[2], score[0], score[1]))
            return action
        
        else:
            return None




class PlayabilityHintsManager(BaseHintsManager):
    """
    Playability hints manager.
    A hint communicates to every other player which of their cards are playable.
    """
    
    def is_appropriate(self, player_id, action):
        """
        Returns True iff the given hint should be processed by this HintsManager.
        At the moment, it is used in the first turn of 5-player games.
        """
        return self.num_players == 5 and self.k == 4 and action.turn == 0
    
    
    def card_to_hint_type(self, player_id):
        """
        For the given player (different from me) associate to each card the hint type to give
        in order to recognise that card.
        
        Example 1: 4 White, 4 Yellow, 3 Yellow, 2 Red.
            4 White -> White (color)
            4 Yellow -> 4 (number)
            3 Yellow -> 3 (number)
            2 Red -> Red (color)
        
        Example 2: 4 White, 4 White, 4 Yellow, 3 Yellow.
            4 White -> White (color)
            4 White -> 4 (number)
            4 Yellow -> Yellow (color)
            3 Yellow -> 3 (number)
        
        Ideally, each card should be associated to a different value (if this is not possible,
        then some card cannot be selected and is associated to the hint type None).
        All the player must agree on the association.
        Ideally it would be better when the chosen value is unique in the hand (in this way, the owner
        can infer the card even without knowing the association). If the value is not unique, the owner
        will not be able to decode the hint. [Not implemented yet]
        Notice that the above example do not need to match the behaviour of this method.
        """
        
        assert player_id != self.id
        
        # create bipartite graph
        G = nx.Graph()
        G.add_nodes_from(xrange(self.k), bipartite = 0)
        G.add_nodes_from(
                [(Action.COLOR, color) for color in Card.COLORS],
                bipartite = 1
            )
        G.add_nodes_from(
                [(Action.NUMBER, number) for number in xrange(1, Card.NUM_NUMBERS + 1)],
                bipartite = 1
            )
        
        for (card_pos, card) in enumerate(self.hands[player_id]):
            G.add_edge(card_pos, (Action.COLOR, card.color))
            G.add_edge(card_pos, (Action.NUMBER, card.number))
        
        left, right = nx.bipartite.sets(G)
        
        # compute matching (hoping that it is deterministic)
        matching = nx.bipartite.maximum_matching(G)
        # TODO: prefer unique values (see documentation above)
        
        return [matching[card_pos] for card_pos in xrange(self.k)]
        
    
    
    def receive_hint(self, player_id, action):
        """
        Receive hint given by player_id.
        """
        
        super(PlayabilityHintsManager, self).receive_hint(player_id, action)
    
    
    def get_hint(self):
        """
        Compute hint to give.
        """
        if self.num_players != 5:
            return None
        
        # give this kind of hint only in 5-player games with 4 cards per player
        assert self.num_players == 5 and self.k == 4
        
        # TODO: continue
        print self.card_to_hint_type(1)
        





