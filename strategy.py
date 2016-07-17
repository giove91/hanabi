#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import itertools
import copy

from action import Action
from card import Card, deck


class Knowledge:
    def __init__(self, color=False, number=False):
        self.color = color
        self.number = number
    
    def __repr__(self):
        return ("C" if self.color else "-") + ("N" if self.number else "-")
    
    
    def knows(self, number_hint=False):
        if number_hint:
            return self.number
        else:
            return self.color



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


    def choose_card(self, player_id, target_id, turn, number_hint):
        # choose which of the target's cards receive a hint from the current player in the given turn
        possible_cards = [card_pos for (card_pos, kn) in enumerate(self.knowledge[target_id]) if not (kn.number if number_hint else kn.color)]
        
        if len(possible_cards) == 0:
            # do not give hints
            return None
        
        # TODO: forse usare un vero hash
        n = turn * 11**3 + (1 if number_hint else 0) * 119 + player_id * 11 + target_id
        
        return possible_cards[n % len(possible_cards)]
    
    
    def choose_all_cards(self, player_id, turn, number_hint):
        # choose all cards that receive hints from the given player
        return {target_id: self.choose_card(player_id, target_id, turn, number_hint) for target_id in xrange(self.num_players) if target_id != player_id and self.choose_card(player_id, target_id, turn, number_hint) is not None}
    
    
    def infer_playable_cards(self, player_id, action):
        """
        From the choice made by the hinter (give hint on color or number), infer something
        about the playability of my cards.
        Here it is important that:
        - playability of a card depends only on things that everyone sees;
        - the choice of the type of hint (color/number) is primarily based on the number of playable cards.
        Call this function before receive_hint(), i.e. before knowledge is updated.
        """
        number_hint = action.number_hint
        cards_pos = self.choose_all_cards(player_id, action.turn, number_hint)
        alternative_cards_pos = self.choose_all_cards(player_id, action.turn, not number_hint)
        
        if self.id not in cards_pos or self.id not in alternative_cards_pos:
            # I already knew about one of the two cards
            return None

        if action.player_id == self.id:
            # the hint was given to me, so I haven't enough information to infer something
            return None
        
        
        if number_hint:
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
        
        
        involved_cards = [hand[cards_pos[i]] for (i, hand) in self.strategy.hands.iteritems() if i != player_id and i in cards_pos] + [self.strategy.hands[action.player_id][card_pos] for card_pos in action.hinted_card_pos]
        
        involved_cards = list(set(involved_cards))
        my_card_pos = cards_pos[self.id]
        num_playable = sum(1 for card in involved_cards if card.playable(self.strategy.board))
        
        alternative_involved_cards = [hand[alternative_cards_pos[i]] for (i, hand) in self.strategy.hands.iteritems() if i != player_id and i in alternative_cards_pos]
        alternative_my_card_pos = alternative_cards_pos[self.id]
        alternative_num_playable = sum(1 for card in alternative_involved_cards if card.playable(self.strategy.board))
        
        # self.log("Num playable: %d, %d" % (num_playable, alternative_num_playable))
        # self.log(involved_cards.__repr__() + " " + my_card_pos.__repr__())
        # self.log(alternative_involved_cards.__repr__() + " " + alternative_my_card_pos.__repr__())
        
        if alternative_num_playable > num_playable:
            assert alternative_num_playable == num_playable + 1
            # found a playable card and a non-playable card!
            self.log("found playable card (%d) and non-playable card (%d)" % (my_card_pos, alternative_my_card_pos))
            return my_card_pos, alternative_my_card_pos
        
        

    def receive_hint(self, player_id, action):
        """
        Decode hint given by someone else (not necessarily directly to me).
        """
        number_hint = action.number_hint
        cards_pos = self.choose_all_cards(player_id, action.turn, number_hint)
        
        # update knowledge
        for (target_id, card_pos) in cards_pos.iteritems():
            kn = self.knowledge[target_id][card_pos]
            if number_hint:
                kn.number = True
            else:
                kn.color = True
        
        for card_pos in action.hinted_card_pos:
            kn = self.knowledge[action.player_id][card_pos]
            if number_hint:
                kn.number = True
            else:
                kn.color = True
        
        # decode my hint
        if self.id in cards_pos:
            n = action.number if number_hint else self.COLORS_TO_NUMBERS[action.color]
            my_card_pos = cards_pos[self.id]
            modulo = Card.NUM_NUMBERS if number_hint else Card.NUM_COLORS
            
            involved_cards = [hand[cards_pos[i]] for (i, hand) in self.strategy.hands.iteritems() if i != player_id and i in cards_pos]
            
            m = sum(card.number if number_hint else self.COLORS_TO_NUMBERS[card.color] for card in involved_cards)
            my_value = (n - m) % modulo
            
            number = my_value if number_hint else None
            if number == 0:
                number = 5
            color = Card.COLORS[my_value] if not number_hint else None
            
            return my_card_pos, color, number
        
        else:
            # no hint (apparently I already know everything)
            return None
    
    
    
    def compute_hint(self, turn, number_hint):
        """
        Returns the color/number we need to give a hint about.
        """
        cards_pos = self.choose_all_cards(self.id, turn, number_hint)
        
        if len(cards_pos) == 0:
            # the other players already know everything
            return None
        
        # compute sum of visible cards in the given positions
        modulo = Card.NUM_NUMBERS if number_hint else Card.NUM_COLORS
        involved_cards = [hand[cards_pos[i]] for (i, hand) in self.strategy.hands.iteritems() if i in cards_pos]
        m = sum(card.number if number_hint else self.COLORS_TO_NUMBERS[card.color] for card in involved_cards) % modulo
        
        number = m if number_hint else None
        if number == 0:
            number = 5
        color = Card.COLORS[m] if not number_hint else None
        
        return color, number
        
    
    def get_best_hint(self, turn):
        """
        Choose the best hint to give, if any.
        """
        # try the two possible number_hint values
        possibilities = {False: None, True: None}   # possibilities for number_hint, with results
        
        for number_hint in [False, True]:
            # compute which cards would be involved in this indirect hint
            cards_pos = self.choose_all_cards(self.id, turn, number_hint)
            involved_cards = [self.strategy.hands[i][card_pos] for (i, card_pos) in cards_pos.iteritems()]
            
            res = self.compute_hint(turn, number_hint)
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
                    involved_cards += [card for (card_pos, card) in enumerate(self.strategy.hands[player_id]) if card is not None and card.matches(color=color, number=number) and not self.knowledge[player_id][card_pos].knows(number_hint)]
                    involved_cards = list(set(involved_cards))
                    num_relevant = sum(1 for card in involved_cards if card.relevant(self.strategy.board, self.strategy.full_deck, self.strategy.discard_pile))
                    num_playable = sum(1 for card in involved_cards if card.playable(self.strategy.board))
                    num_useful = sum(1 for card in involved_cards if card.useful(self.strategy.board, self.strategy.full_deck, self.strategy.discard_pile))
                    
                    # self.log("involved cards: " + involved_cards.__repr__())
                    # self.log("there are %d playable, %d relevant, %d useful cards" % (num_playable, num_relevant, num_useful))
                    
                    # TODO: tener conto dei giocatori che eventualmente non giocheranno più
                    # TODO: in generale, gestire in modo più preciso la parte finale della partita (quanto si sa quasi tutto)
                    # Give priority to playable cards, then to relevant cards, then to the number of cards.
                    # WARNING: it is important that the first parameter is the number of playable cards,
                    # because other players obtain information from this.
                    # If the hint doesn't involve any useful card, avoid giving the hint.
                    if num_useful > 0:
                        possibilities[number_hint] = (num_playable, num_relevant, len(involved_cards)), Action(Action.HINT, player_id=player_id, color=color, number=number)
        
        
        # choose between color and number
        possibilities = {a: b for (a,b) in possibilities.iteritems() if b is not None}
        
        if len(possibilities) > 0:
            score, action = sorted(possibilities.itervalues(), key = lambda x: x[0])[-1]
            self.log("giving indirect hint on %d cards with score %d, %d" % (score[2], score[0], score[1]))
            return action
        
        else:
            return None



class Strategy:
    """
    An instance of this class represents a player's strategy.
    It only has the knowledge of that player, and it must make decisions.
    """
    
    def __init__(self, debug=False):
        self.COLORS_TO_NUMBERS = {color: i for (i, color) in enumerate(Card.COLORS)}
        self.debug = debug
    
    
    def initialize(self, id, num_players, k, hands, board, discard_pile):
        # to be called once before the beginning
        self.id = id
        self.num_players = num_players
        self.k = k  # number of cards per hand
        self.hands = hands  # hands of other players
        self.board = board
        self.discard_pile = discard_pile
        
        # store a copy of the full deck
        self.full_deck = deck()
        
        # for each of my card, I store its possibilities
        self.possibilities = [set(self.full_deck) for i in xrange(self.k)]
        
        # remove cards of other players from possibilities
        self.update_possibilities()
        
        # indirect hints manager
        self.hints_manager = HintsManager(num_players, k, id, self)
    
    
    def update(self, hints, lives, my_hand, turn, last_turn, deck_size):
        # to be called every turn
        self.hints = hints
        self.lives = lives
        self.my_hand = my_hand
        self.turn = turn
        self.last_turn = last_turn
        self.deck_size = deck_size
    
    
    def visible_cards(self):
        for card in self.discard_pile:
            yield card
        
        for hand in self.hands.itervalues():
            for card in hand:
                yield card
    
    def update_possibilities(self):
        # update possibilities removing visible cards
        for card in self.visible_cards():
            for p in self.possibilities:
                if card in p:
                    p.remove(card)
    
    def update_possibilities_with_combinations(self):
        # update possibilities examining all combinations of my hand
        # better to do it with only few cards remaining!
        possible_cards = set()
        for p in self.possibilities:
            possible_cards |= p
        
        new_possibilities = [set() for card_pos in xrange(self.k)]
        
        num_cards = len([x for x in self.my_hand if x is not None])
        assert num_cards <= self.k
        
        # cycle over all combinations
        for comb in itertools.permutations(possible_cards, num_cards):
            # construct hand
            hand = copy.copy(self.my_hand)
            i = 0
            for card_pos in xrange(self.k):
                if hand[card_pos] is not None:
                    hand[card_pos] = comb[i]
                    i += 1
            
            # check if this hand is possible
            if all(card is None or card in self.possibilities[card_pos] for (card_pos, card) in enumerate(hand)):
                # this hand is possible
                # self.log("possible hand " + hand.__repr__())
                
                for (card_pos, card) in enumerate(hand):
                    if card is not None:
                        new_possibilities[card_pos].add(card)
        
        self.log("old possibilities " + [len(p) for p in self.possibilities].__repr__())
        self.log("new possibilities " + [len(p) for p in new_possibilities].__repr__())
        
        # update possibilities
        self.possibilities = new_possibilities
    
    
    def next_player_id(self):
        return (self.id + 1) % self.num_players
    
    def other_players_id(self):
        return [i for i in xrange(self.num_players) if i != self.id]
    
    
    def feed_turn(self, player_id, action):
        if action.type in [Action.PLAY, Action.DISCARD]:
            # reset knowledge of the player
            new_card = self.my_hand[action.card_pos] if player_id == self.id else self.hands[player_id][action.card_pos]
            self.hints_manager.reset_knowledge(player_id, action.card_pos, new_card is not None)
            
            if player_id == self.id:
                # check for my new card
                self.possibilities[action.card_pos] = set(self.full_deck) if self.my_hand[action.card_pos] is not None else set()
        
        elif action.type == Action.HINT:
            # someone gave a hint!
            
            if action.player_id == self.id:
                # they gave me a hint!
                
                # process direct hint
                for (i, p) in enumerate(self.possibilities):
                    for card in self.full_deck:
                        if not card.matches_hint(action, i) and card in p:
                            # self.log("removing card " + card.__repr__() + " from position %d due to hint" % i)
                            p.remove(card)
            
            else:
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
                                    # self.log("removing card " + card.__repr__() + " from position %d due to hint skip" % i)
                                    p.remove(card)
                
            # infer playability of some cards, from the type of the given hint
            res = self.hints_manager.infer_playable_cards(player_id, action)
            
            if res is not None:
                # found a playable and a non-playable card
                playable, non_playable = res
                for card in self.full_deck:
                    if card.playable(self.board) and card in self.possibilities[non_playable]:
                        # self.log("removing " + card.__repr__() + " from position %d" % non_playable)
                        self.possibilities[non_playable].remove(card)
                    elif not card.playable(self.board) and card in self.possibilities[playable]:
                        # self.log("removing " + card.__repr__() + " from position %d" % playable)
                        self.possibilities[playable].remove(card)
            
            # process indirect hint
            res = self.hints_manager.receive_hint(player_id, action)
            
            if res is not None:
                card_pos, color, number = res
                # self.log("thanks to indirect hint, understood that card %d has " % card_pos + ("number %d" % number if action.number_hint else "color %s" % color))
            
                p = self.possibilities[card_pos]
                for card in self.full_deck:
                    if not card.matches(color=color, number=number) and card in p:
                        p.remove(card)
        
        
        # update possibilities with visible cards
        self.update_possibilities()
        
        # print knowledge
        if self.debug and self.id == self.num_players-1:
            self.hints_manager.print_knowledge()
    
    
    def get_best_discard(self):
        """
        Choose the best card to be discarded.
        """
        # first see if I can be sure to discard a useless card
        for (card_pos, p) in enumerate(self.possibilities):
            if len(p) > 0 and all(not card.useful(self.board, self.full_deck, self.discard_pile) for card in p):
                self.log("discard useless card")
                return card_pos
        
        # Try to avoid cards that are likely relevant, then choose cards that are more likely useless
        tolerance = 1e-3
        best_cards_pos = []
        best_relevant_ratio = 1.0
        for (card_pos, p) in enumerate(self.possibilities):
            if len(p) > 0:
                num_relevant = sum(1 for card in p if card.relevant(self.board, self.full_deck, self.discard_pile))
                relevant_ratio = float(num_relevant) / len(p)
                
                num_useful = sum(1 for card in p if card.useful(self.board, self.full_deck, self.discard_pile))
                useful_ratio = float(num_useful) / len(p)
                
                if relevant_ratio < best_relevant_ratio - tolerance:
                    # better ratio found
                    best_cards_pos, best_relevant_ratio = [], relevant_ratio
                
                if relevant_ratio < best_relevant_ratio + tolerance:
                    # add this card to the possibilities
                    best_cards_pos.append((useful_ratio, card_pos))
        
        assert len(best_cards_pos) > 0
        useful_ratio, card_pos = sorted(best_cards_pos)[0]
        
        self.log("discard a card (relevant ratio ~%.3f, useful ratio %.3f)" % (best_relevant_ratio, useful_ratio))
        return card_pos
    
    
    def get_best_play(self):
        """
        Choose the best card to play.
        """
        # TODO: preferire carte basse e i 5 (perché danno indizi)
        best_card_pos = None
        best_avg_num_playable = -1.0
        for (card_pos, p) in enumerate(self.possibilities):
            if all(card.playable(self.board) for card in p) and len(p) > 0:
                # the card in this position is surely playable!
                
                # how many cards of the other players become playable, on average?
                num_playable = []
                for card in p:
                    fake_board = copy.copy(self.board)
                    fake_board[card.color] += 1
                    num_playable.append(sum(1 for (player_id, hand) in self.hands.iteritems() for c in hand if c is not None and c.playable(fake_board)))
                
                avg_num_playable = float(sum(num_playable)) / len(num_playable)
                
                if avg_num_playable > best_avg_num_playable:
                    best_card_pos, best_avg_num_playable = card_pos, avg_num_playable
        
        if best_card_pos is not None:
            self.log("playing card in position %d gives %f playable cards on average" % (best_card_pos, best_avg_num_playable))
        return best_card_pos
    
    
    
    def get_turn_action(self):
        # update possibilities checking all combinations
        if self.deck_size < 10:
            self.update_possibilities_with_combinations()
        
        # check for playable cards in my hand
        card_pos = self.get_best_play()
        if card_pos is not None:
            # play the card
            return Action(Action.PLAY, card_pos=card_pos)
        
        
        if self.hints == 0:
            # discard card
            return Action(Action.DISCARD, card_pos=self.get_best_discard())
        
        # TODO: forse è meglio scartare se quello dopo ha carte rilevanti di cui non è a conoscenza, o se ha solo carte rilevanti
        
        # try to give indirect hint
        best_hint = self.hints_manager.get_best_hint(self.turn)
        if best_hint is not None:
            return best_hint
        
        else:
            # failed to give indirect hint
            # discard card
            self.log("failed to give a hint")
            return Action(Action.DISCARD, card_pos=self.get_best_discard())



    
    def log(self, message):
        if self.debug:
            print "Player %d: %s" % (self.id, message)




