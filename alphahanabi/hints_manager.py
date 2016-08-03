#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import copy

sys.path.append("..") 


from action import Action, PlayAction, DiscardAction, HintAction
from card import Card



class Knowledge:
    def __init__(self, color=False, number=False, one=False):
        self.color = color
        self.number = number
        self.one = one  # knowledge about this card being a (likely) playable one
    
    def __repr__(self):
        return ("C" if self.color else "-") + ("N" if self.number else "-") + ("O" if self.one else "-")
    
    
    def knows(self, hint_type):
        assert hint_type in Action.HINT_TYPES
        if hint_type == Action.COLOR:
            return self.color
        else:
            return self.number



class HintsManager:
    def __init__(self, num_players, k, id, strategy):
        self.num_players = num_players
        self.k = k
        self.id = id    # my player id
        self.strategy = strategy    # my strategy object
        self.knowledge = [[Knowledge(color=False, number=False) for j in xrange(k)] for i in xrange(num_players)]
        
        self.COLORS_TO_NUMBERS = {color: i for (i, color) in enumerate(Card.COLORS)}
    
    
    def log(self, message):
        self.strategy.log(message)
    
    def is_first_round(self):
        return self.strategy.turn < self.num_players


    def reset_knowledge(self, player_id, card_pos, new_card_exists):
        self.knowledge[player_id][card_pos] = Knowledge(False, False) if new_card_exists else Knowledge(True, True)
    
    def print_knowledge(self):
        print "Knowledge"
        for i in xrange(self.num_players):
            print "Player %d:" % i,
            for card_pos in xrange(self.k):
                print self.knowledge[i][card_pos],
            print
        print

    
    def shift(self, turn):
        # a "random" shift in the hint
        return turn + turn / self.num_players
    
    
    def is_duplicate(self, card):
        # says if the given card is owned by some player who knows everything about it
        
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
        # choose which of the target's cards receive a hint from the current player in the given turn
        possible_cards = [card_pos for (card_pos, kn) in enumerate(self.knowledge[target_id]) if not (kn.color if hint_type == Action.COLOR else kn.number)]
        
        if len(possible_cards) == 0:
            # do not give hints
            return None
        
        # TODO: forse usare un vero hash
        n = turn * 11**3 + (0 if hint_type == Action.COLOR else 1) * 119 + player_id * 11 + target_id
        
        return possible_cards[n % len(possible_cards)]
    
    
    def choose_all_cards(self, player_id, turn, hint_type):
        # choose all cards that receive hints from the given player
        return {target_id: self.choose_card(player_id, target_id, turn, hint_type) for target_id in xrange(self.num_players) if target_id != player_id and self.choose_card(player_id, target_id, turn, hint_type) is not None}
    
    
    def infer_playable_cards(self, player_id, action):
        """
        From the choice made by the hinter (give hint on color or number), infer something
        about the playability of my cards.
        Here it is important that:
        - playability of a card depends only on things that everyone sees;
        - the choice of the type of hint (color/number) is primarily based on the number of playable cards.
        Call this function before receive_hint(), i.e. before knowledge is updated.
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
        
        """
        if self.is_first_round():
            # In the first round hints on 1s are natural, so it's better not to infer anything.
            return None
        """
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
        
        

    def receive_hint(self, player_id, action):
        """
        Decode hint given by someone else (not necessarily directly to me).
        """
        hint_type = action.hint_type
        cards_pos = self.choose_all_cards(player_id, action.turn, hint_type)
        
        """
        if self.is_first_round() and action.number == 1:
            # In the first round, hints on 1s are natural.
            
            # Update knowledge.
            for card_pos in action.cards_pos:
                kn = self.knowledge[action.player_id][card_pos]
                kn.number = True
                kn.one = True
                kn.color = True # he doesn't know the color, but he will play the one anyway
            
            return None
        """
    
        # update knowledge
        for (target_id, card_pos) in cards_pos.iteritems():
            kn = self.knowledge[target_id][card_pos]
            if hint_type == Action.COLOR:
                kn.color = True
            else:
                kn.number = True
        
        for card_pos in action.cards_pos:
            kn = self.knowledge[action.player_id][card_pos]
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
            
            number = my_value if hint_type == Action.NUMBER else None
            if number == 0:
                number = 5
            color = Card.COLORS[my_value] if hint_type == Action.COLOR else None
            
            return my_card_pos, color, number
        
        else:
            # no hint (apparently I already know everything)
            return None
    
    
    
    def compute_hint_value(self, turn, hint_type):
        """
        Returns the color/number we need to give a hint about.
        """
        cards_pos = self.choose_all_cards(self.id, turn, hint_type)
        
        if len(cards_pos) == 0:
            # the other players already know everything
            return None
        
        # compute sum of visible cards in the given positions
        modulo = Card.NUM_NUMBERS if hint_type == Action.NUMBER else Card.NUM_COLORS
        involved_cards = [hand[cards_pos[i]] for (i, hand) in self.strategy.hands.iteritems() if i in cards_pos]
        m = sum(card.number if hint_type == Action.NUMBER else self.COLORS_TO_NUMBERS[card.color] for card in involved_cards) + self.shift(turn)
        m %= modulo
        
        number = m if hint_type == Action.NUMBER else None
        if number == 0:
            number = 5
        color = Card.COLORS[m] if hint_type == Action.COLOR else None
        
        return color, number
    
    
    def get_best_hint_on_ones(self):
        """
        Returns the best natural hint on playable ones.
        """
        assert all(not kn.one for kn in self.knowledge[self.id])    # If I knew some 1, I should have played it.
        
        virtually_played = set()
        
        for color in Card.COLORS:
            if self.strategy.board[color] >= 1:
                # a 1 of that color was already played
                virtually_played.add(color)
        
        for (player_id, knowledge) in enumerate(self.knowledge):
            if player_id != self.id:
                for (card_pos, kn) in enumerate(knowledge):
                    if kn.one:
                        # this player knows about a 1
                        virtually_played.add(self.strategy.hands[player_id][card_pos].color)
        
        self.log("virtually played ones are %r" % virtually_played)
        
        # analyze hands of other players
        new_colors = {}
        best = 0
        best_player_id = None
        for (player_id, hand) in self.strategy.hands.iteritems():
            owned = [card.color for (card_pos, card) in enumerate(hand) if card.number == 1 and not self.knowledge[player_id][card_pos].one]
            
            if len(owned) == len(set(owned)) and all(color not in virtually_played for color in owned):
                # this player is suitable for my hint!
                new_colors[player_id] = owned
                if len(owned) > best:
                    best, best_player_id = len(owned), player_id
        
        # TODO: migliorare rispetto alla scelta greedy
        if best_player_id is not None and best >= 3:
            self.log("give natural hints on 1s to player %d (%d cards)" % (best_player_id, best))
            return HintAction(player_id=best_player_id, number=1)
        
        else:
            return None

    
    def get_best_hint(self):
        """
        Choose the best hint to give, if any.
        """
        """
        if self.is_first_round():
            # Try to give a natural hint on 1s.
            hint_action = self.get_best_hint_on_ones()
            if hint_action is not None:
                return hint_action
        """
        # try the two possible hint_type values
        possibilities = {hint_type: None for hint_type in Action.HINT_TYPES}
        
        for hint_type in Action.HINT_TYPES:
            # compute which cards would be involved in this indirect hint
            cards_pos = self.choose_all_cards(self.id, self.strategy.turn, hint_type)
            involved_cards = [self.strategy.hands[i][card_pos] for (i, card_pos) in cards_pos.iteritems()]
            
            res = self.compute_hint_value(self.strategy.turn, hint_type)
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
        """
        if self.is_first_round():
            # Hints on 1s are natural in the first round.
            if possibilities[True] is not None and possibilities[True][1].number == 1:
                possibilities[True] = None
        """
        
        # choose between color and number
        possibilities = {a: b for (a,b) in possibilities.iteritems() if b is not None}
        

        if len(possibilities) > 0:
            score, action = sorted(possibilities.itervalues(), key = lambda x: x[0])[-1]
            self.log("giving indirect hint on %d cards with score %d, %d" % (score[2], score[0], score[1]))
            return action
        
        else:
            return None


