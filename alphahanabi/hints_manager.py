#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import copy

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
    
    
    def is_usable(self, hinter_id):
        """
        Check that it is possible to pass all the information.
        """
        return True
    
    
    def receive_hint(self, player_id, action):
        """
        Receive hint given by player_id and update knowledge.
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
    


class SumBasedHintsManager(BaseHintsManager):
    """
    A HintManager which is based on the following idea.
    Associate an integer (between 0 and M) to any possible hand. Send the sum of the integers
    associated to the hands of the other players, modulo M. Then each other player can decode
    its integer by difference, seeing the hands of the other players.
    
    With this implementation, M = number of cards of the other players. The integer is encoded
    in the choice of the card to give a hint about.
    """
    
    def hash(self, hand):
        """
        The hash of the hand that we want to communicate.
        Must be an integer.
        """
        # To be overloaded by child classes.
        raise NotImplementedError
    
    
    def hash_range(self):
        """
        Return H such that 0 <= hash < H.
        """
        # To be overloaded by child classes.
        raise NotImplementedError
    
    
    def process_hash(self, x):
        """
        Process the given hash of my hand, passed through a hint.
        Optionally, return data to be used by update_knowledge.
        """
        # To be overloaded by child classes.
        raise NotImplementedError
    
    
    def update_knowledge(self, hinter_id, data=None):
        """
        Update knowledge after a hint has been given.
        Optionally, get data from process_hash.
        """
        # To be overloaded by child classes.
        raise NotImplementedError
    
    
    
    def is_usable(self, hinter_id):
        """
        Check if there are enough cards to pass all the information.
        """
        return self.hash_range() <= self.modulo(hinter_id)
    
    
    def compute_hash_sum(self, hinter_id):
        """
        Compute the sum of the hashes of the hands of the other players, excluding the hinter and myself.
        """
        res = 0
        for (player_id, hand) in self.hands.iteritems():
            if player_id != hinter_id:
                h = self.hash(hand)
                assert 0 <= h < self.modulo(hinter_id)
                res += self.hash(hand)
        return res
    
    
    def cards_to_hints(self, player_id):
        """
        For the given player (different from me) return a matching between each card and the hint (type and value)
        to give in order to recognise that card.
        
        Example 1: 4 White, 4 Yellow, 3 Yellow, 2 Red.
            4 White -> White (color)
            4 Yellow -> 4 (number)
            3 Yellow -> 3 (number)
            2 Red -> Red (color)
        
        Example 2: 4 White, 4 White, 4 Yellow, 3 Yellow.
            4 White -> White (color)
            4 White -> None (both 4 and White would mean other cards)
            4 Yellow -> Yellow (color)
            3 Yellow -> 3 (number)
        
        In case a value is not unique in the hand, color means leftmost card and number means rightmost card.
        """
        
        assert player_id != self.id
        hand = self.hands[player_id]
        
        matching = {}
        
        # analyze hints on color
        for color in Card.COLORS:
            cards_pos = [card_pos for (card_pos, card) in enumerate(hand) if card.matches(color=color)]
            if len(cards_pos) > 0:
                # pick the leftmost
                card_pos = min(cards_pos)
                matching[card_pos] = (Action.COLOR, color)
                matching[(Action.COLOR, color)] = card_pos
        
        # analyze hints on numbers
        for number in xrange(1, Card.NUM_NUMBERS+1):
            cards_pos = [card_pos for (card_pos, card) in enumerate(hand) if card.matches(number=number)]
            if len(cards_pos) > 0:
                # pick the rightmost
                card_pos = max(cards_pos)
                matching[card_pos] = (Action.NUMBER, number)
                matching[(Action.NUMBER, number)] = card_pos
        
        return matching
    
    
    def hint_to_card(self, action):
        """
        From the hint, understand the important card.
        This is the inverse of cards_to_hints.
        """
        if action.hint_type == Action.COLOR:
            # pick the leftmost card
            return min(action.cards_pos)
        else:
            # pick the rightmost card
            return max(action.cards_pos)
        
    
    def relevant_cards(self, hinter_id):
        """
        Matching between integers and cards of players other than the hinter, in the form (player_id, card_pos).
        For example:
            0: (0, 1)   # player 0, card 1
            1: (0, 2)   # player 0, card 2
            ...
            (0, 1): 0
            (0, 2): 1
            ...
        """
        matching = {}
        counter = 0
        
        for player_id in xrange(self.num_players):
            if player_id == hinter_id:
                # skip the hinter
                continue
            
            if player_id == self.id:
                hand = self.my_hand
            else:
                hand = self.hands[player_id]
            
            for (card_pos, card) in enumerate(hand):
                if card is not None:
                    matching[counter] = (player_id, card_pos)
                    matching[(player_id, card_pos)] = counter
                    counter += 1
        
        return matching
    
    
    def modulo(self, hinter_id):
        """
        Returns the number of different choices for the integer that needs to be communicated.
        """
        # for our protocol, such number is the number of cards of the other players
        return len(self.relevant_cards(hinter_id)) / 2
    
    
    def get_hint(self):
        """
        Compute hint to give.
        """
        x = self.compute_hash_sum(self.id) % self.modulo(self.id)
        self.log("communicate message %d" % x)
        
        relevant_cards = self.relevant_cards(self.id)
        player_id, card_pos = relevant_cards[x]
        
        matching = self.cards_to_hints(player_id)
        
        if card_pos in matching:
            hint_type, value = matching[card_pos]
            return HintAction(player_id=player_id, hint_type=hint_type, value=value)
        
        else:
            # unable to give hint on that card
            return None
    
    
    def hint_to_integer(self, hinter_id, action):
        """
        Decode an HintAction and get my integer.
        """
        # this only makes sense if I am not the hinter
        assert self.id != hinter_id
        
        # compute passed integer
        player_id = action.player_id
        card_pos = self.hint_to_card(action)
        
        relevant_cards = self.relevant_cards(hinter_id)
        x = relevant_cards[(player_id, card_pos)]
        
        self.log("received message %d" % x)
        
        # compute difference with other hashes
        y = (x - self.compute_hash_sum(hinter_id)) % self.modulo(hinter_id)
        
        return y
    
    
    def receive_hint(self, player_id, action):
        if self.id != player_id:
            # I am not the hinter
            x = self.hint_to_integer(player_id, action)
            self.log("the hash of my hand is %d" % x)
            data = self.process_hash(x)
        else:
            data = None
        
        self.update_knowledge(player_id, data)
        
        super(SumBasedHintsManager, self).receive_hint(player_id, action)
        



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
        """
        Receive hint given by player_id and update knowledge.
        """
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





class PlayabilityHintsManager(SumBasedHintsManager):
    """
    Playability hints manager.
    A hint communicates to every other player which of their cards are playable.
    """
    
    def hash(self, hand):
        """
        This hash encodes which cards are playable (as a binary integer).
        """
        string = "".join(["1" if card.playable(self.board) else "0" for card in hand])
        return int(string, 2)
    
    
    def hash_range(self):
        """
        Return H such that 0 <= hash < H.
        """
        return 2 ** self.k
    
    
    def process_hash(self, x):
        """
        Process the given hash of my hand, passed through a hint.
        """
        assert 0 <= x < 2 ** self.k
        string = str(bin(x))[2:].zfill(self.k)
        assert len(string) == self.k
        playable_list = [True if c == "1" else False for c in string]
        self.log("received playable string %s" % string)
        
        # update possibilities
        for (card_pos, playable) in enumerate(playable_list):
            for card in self.full_deck:
                p = self.possibilities[card_pos]
                if card in p and (playable and not card.playable(self.board) or not playable and card.playable(self.board)):
                    # self.log("removing %r from position %d" % (card, card_pos))
                    p.remove(card)
                
                # TODO: take into account duplicate cards! (should not be considered playable)
                
                """
                if card.playable(self.board) and card in self.possibilities[non_playable] and not self.is_duplicate(card):
                    # self.log("removing %r from position %d" % (card, non_playable))
                    self.possibilities[non_playable].remove(card)
                elif not card.playable(self.board) and card in self.possibilities[playable] and not self.is_duplicate(card):
                    # self.log("removing %r from position %d" % (card, playable))
                    self.possibilities[playable].remove(card)
                """
        
        # return data for update_knowledge
        return playable_list
    
    
    def update_knowledge(self, hinter_id, data):
        """
        Update knowledge after a hint has been given.
        """
        if hinter_id != self.id:
            # update my knowledge
            playable_list = data
            for (card_pos, playable) in enumerate(playable_list):
                kn = self.knowledge[self.id][card_pos]
                if playable:
                    kn.playable = True
                else:
                    kn.non_playable = True
        
        # update knowledge of players different by me and the hinter
        for (p_id, hand) in self.hands.iteritems():
            if p_id == hinter_id:
                # skip the hinter
                continue
            
            for (card_pos, card) in enumerate(hand):
                kn = self.knowledge[p_id][card_pos]
                if card.playable(self.board):
                    kn.playable = True
                else:
                    kn.non_playable = True
    

