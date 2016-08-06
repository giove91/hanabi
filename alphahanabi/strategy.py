#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import copy

sys.path.append("..") 


from action import Action, PlayAction, DiscardAction, HintAction
from card import Card, deck
from base_strategy import BaseStrategy
from hints_manager import ValueHintsManager




class Knowledge:
    """
    An instance of this class represents what a player knows about a card, as known by everyone.
    """
    
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




class Strategy(BaseStrategy):
    """
    An instance of this class represents a player's strategy.
    It only has the knowledge of that player, and it must make decisions.
    """
    
    def __init__(self, verbose=False):
        self.COLORS_TO_NUMBERS = {color: i for (i, color) in enumerate(Card.COLORS)}
        self.verbose = verbose
    
    
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
        
        # knowledge of all players
        self.knowledge = [[Knowledge(color=False, number=False) for j in xrange(k)] for i in xrange(num_players)]
        
        # hints manager(s)
        self.value_hints_manager = ValueHintsManager(self)
    
    
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
        assert all(len(p) > 0 or self.my_hand[card_pos] is None for (card_pos, p) in enumerate(self.possibilities))
    
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
                # self.log("possible hand %r" % hand)
                
                for (card_pos, card) in enumerate(hand):
                    if card is not None:
                        new_possibilities[card_pos].add(card)
        
        self.log("old possibilities %r" % [len(p) for p in self.possibilities])
        self.log("new possibilities %r" % [len(p) for p in new_possibilities])
        
        # update possibilities (avoid direct assignment to mantain references)
        for (card_pos, p) in enumerate(self.possibilities):
            p &= new_possibilities[card_pos]
    
    
    def next_player_id(self):
        return (self.id + 1) % self.num_players
    
    def other_players_id(self):
        return [i for i in xrange(self.num_players) if i != self.id]
    
    
    def reset_knowledge(self, player_id, card_pos, new_card_exists):
        self.knowledge[player_id][card_pos] = Knowledge(False, False) if new_card_exists else Knowledge(True, True)
        # TODO: forse è meglio mettere False, False per le carte che non esistono
    
    
    def print_knowledge(self):
        print "Knowledge"
        for i in xrange(self.num_players):
            print "Player %d:" % i,
            for card_pos in xrange(self.k):
                print self.knowledge[i][card_pos],
            print
        print

    
    def feed_turn(self, player_id, action):
        assert self.possibilities is self.value_hints_manager.possibilities
        if action.type in [Action.PLAY, Action.DISCARD]:
            # reset knowledge of the player
            new_card = self.my_hand[action.card_pos] if player_id == self.id else self.hands[player_id][action.card_pos]
            self.reset_knowledge(player_id, action.card_pos, new_card is not None)
            
            if player_id == self.id:
                # check for my new card
                self.possibilities[action.card_pos] = set(self.full_deck) if self.my_hand[action.card_pos] is not None else set()
        
        elif action.type == Action.HINT:
            # someone gave a hint!
            # use the right hints manager to process it
            self.value_hints_manager.receive_hint(player_id, action)
        
        # update possibilities with visible cards
        self.update_possibilities()
        
        # print knowledge
        if self.verbose and self.id == self.num_players-1:
            self.print_knowledge()

    
    
    def get_best_discard(self):
        """
        Choose the best card to be discarded.
        """
        # first see if I can be sure to discard a useless card
        for (card_pos, p) in enumerate(self.possibilities):
            if len(p) > 0 and all(not card.useful(self.board, self.full_deck, self.discard_pile) for card in p):
                self.log("discard useless card")
                return card_pos, 0.0, 0.0
        
        # Try to avoid cards that are (on average) more relevant, then choose cards that are (on average) less useful
        tolerance = 1e-3
        best_cards_pos = []
        best_relevant_ratio = 1.0
        
        WEIGHT = {number: Card.NUM_NUMBERS + 1 - number for number in xrange(1, Card.NUM_NUMBERS + 1)}
        best_relevant_weight = max(WEIGHT.itervalues())
        
        for (card_pos, p) in enumerate(self.possibilities):
            if len(p) > 0:
                num_relevant = sum(1 for card in p if card.relevant(self.board, self.full_deck, self.discard_pile))
                relevant_weight_sum = sum(WEIGHT[card.number] for card in p if card.relevant(self.board, self.full_deck, self.discard_pile))
                
                relevant_ratio = float(num_relevant) / len(p)
                relevant_weight = float(relevant_weight_sum) / len(p)
                
                num_useful = sum(1 for card in p if card.useful(self.board, self.full_deck, self.discard_pile))
                useful_weight_sum = sum(WEIGHT[card.number] for card in p if card.useful(self.board, self.full_deck, self.discard_pile))
                useful_ratio = float(num_useful) / len(p)
                useful_weight = float(useful_weight_sum) / len(p)
                
                
                if relevant_weight < best_relevant_weight - tolerance:
                    # better weight found
                    best_cards_pos, best_relevant_weight, = [], relevant_weight
                
                if relevant_weight < best_relevant_weight + tolerance:
                    # add this card to the possibilities
                    # self.log("new possibility for discard, pos %d, cards %r, useful weight %.3f" % (card_pos, p, useful_weight))
                    best_cards_pos.append((useful_weight, card_pos))
        
        assert len(best_cards_pos) > 0
        useful_weight, card_pos = sorted(best_cards_pos)[0]
        
        self.log("discard a card (relevant weight ~%.3f, useful weight %.3f)" % (best_relevant_weight, useful_weight))
        return card_pos, relevant_weight, useful_weight
    
    
    def get_best_play(self):
        """
        Choose the best card to play.
        """
        """
        # If I know a 1 is playable, I play it.
        for (card_pos, kn) in enumerate(self.knowledge[self.id]):
            if kn.one:
                self.log("playing 1 in position %d" % card_pos)
                return card_pos
        """
        # I prefer playing cards that allow a higher number of playable cards.
        # In case of tie, I prefer (in this order): NUM_NUMBERS, 1, 2, 3, ..., NUM_NUMBERS-1 (and I give weights accordingly).
        
        WEIGHT = {number: Card.NUM_NUMBERS - number for number in xrange(1, Card.NUM_NUMBERS)}
        WEIGHT[Card.NUM_NUMBERS] = Card.NUM_NUMBERS
        
        tolerance = 1e-3
        best_card_pos = None
        best_avg_num_playable = -1.0    # average number of other playable cards, after my play
        best_avg_weight = 0.0           # average weight (in the sense above)
        for (card_pos, p) in enumerate(self.possibilities):
            if all(card.playable(self.board) for card in p) and len(p) > 0:
                # check that the cards do not overlap with playable 1s of other players
                
                # self.log("checking if card in position %d does not overlap..." % card_pos)
                good = True
                for (player_id, knowledge) in enumerate(self.knowledge):
                    if player_id != self.id:
                        for (c_pos, kn) in enumerate(knowledge):
                            # self.log("[position %d] checking player %d, card in position %d, kn.one %r, card %r, card in p %r" % (card_pos, player_id, c_pos, kn.one, self.hands[player_id][c_pos], self.hands[player_id][c_pos] in p))
                            if kn.one and any(card.equals(self.hands[player_id][c_pos]) for card in p):
                                good = False
                
                if not good:
                    # some of my possible cards overlap with a playable one of someone else
                    self.log("some of my possible cards overlap with a playable 1 of someone else")
                    continue
                
                # the card in this position is surely playable!
                
                # how many cards of the other players become playable, on average?
                num_playable = []
                for card in p:
                    fake_board = copy.copy(self.board)
                    fake_board[card.color] += 1
                    num_playable.append(sum(1 for (player_id, hand) in self.hands.iteritems() for c in hand if c is not None and c.playable(fake_board)))
                
                avg_num_playable = float(sum(num_playable)) / len(num_playable)
                
                avg_weight = float(sum(WEIGHT[card.number] for card in p)) / len(p)
                
                if avg_num_playable > best_avg_num_playable + tolerance or avg_num_playable > best_avg_num_playable - tolerance and avg_weight > best_avg_weight:
                    self.log("update card to be played, pos %d, score %f, %f" % (card_pos, avg_num_playable, avg_weight))
                    best_card_pos, best_avg_num_playable, best_avg_weight = card_pos, avg_num_playable, avg_weight
        
        if best_card_pos is not None:
            self.log("playing card in position %d gives %f playable cards on average and weight %f" % (best_card_pos, best_avg_num_playable, best_avg_weight))
        return best_card_pos
    
    
    def get_best_play_last_round(self):
        """
        Choose the best card to play in the last round of the game.
        The players know almost everything, and it is reasonable to examine all the possibilities.
        """
        best_card_pos = None
        best_avg_score = 0.0
        
        for (card_pos, p) in enumerate(self.possibilities):
            if any(card.playable(self.board) for card in p):
                # at least a card (among the possible ones in this position) is playable
                
                obtained_scores = []
                
                for card in p:
                    # simulate what happens if I play this card
                    best_score = 0
                    for comb in itertools.product(range(self.k), repeat = self.last_turn - self.turn):
                        turn = self.turn
                        board = copy.copy(self.board)
                        player_id = self.id
                        lives = self.lives
                        
                        if card.playable(board):
                            board[card.color] += 1
                        else:
                            lives -= 1
                        
                        if lives >= 1:
                            # simulate other players
                            for (i, c_pos) in enumerate(comb):
                                turn += 1
                                player_id = (player_id + 1) % self.num_players
                                
                                # this player plays the card in position c_pos
                                c = self.hands[player_id][c_pos]
                                if c.playable(board):
                                    board[c.color] += 1
                        
                        score = sum(board.itervalues())
                        best_score = max(score, best_score) # assume that the other players play optimally! :)
                    
                    # self.log("simulation for card %r in position %d gives best score %d" % (card, card_pos, best_score))
                    obtained_scores.append(best_score)
                
                avg_score = float(sum(obtained_scores)) / len(obtained_scores)
                self.log("playing card in position %d gives an average score of %.3f" % (card_pos, avg_score))
                
                if avg_score > best_avg_score:
                    best_card_pos, best_avg_score = card_pos, avg_score
        
        if best_card_pos is not None:
            self.log("playing card in position %d is the best choice" % best_card_pos)
            return best_card_pos

    
    def get_turn_action(self):
        # update possibilities checking all combinations
        if self.deck_size < 10:
            self.update_possibilities_with_combinations()
        
        # if this is the last round, play accordingly
        if self.last_turn is not None:
            card_pos = self.get_best_play_last_round()
            if card_pos is not None:
                # play the card
                return PlayAction(card_pos=card_pos)
        
        else:
            # check for playable cards in my hand
            card_pos = self.get_best_play()
            if card_pos is not None:
                # play the card
                return PlayAction(card_pos=card_pos)
        
        
            
        
        if self.hints == 0:
            # discard card
            return DiscardAction(card_pos=self.get_best_discard()[0])
        
        
        if self.hints <= 1 and self.deck_size >= 2:
            # better to discard if the next player has many important cards
            self.log("there is only one hint, should I discard?")
            card_pos, relevant_weight, useful_weight = self.get_best_discard()
            tolerance = 1e-3
            
            # TODO: se il giocatore successivo ha almeno una carta giocabile di cui è a conoscenza,
            # controllare quello dopo ancora (e così via)
            
            if useful_weight < tolerance:
                # discard is surely good
                return DiscardAction(card_pos=card_pos)
            
            elif all(card.relevant(self.board, self.full_deck, self.discard_pile) for card in self.hands[self.next_player_id()]):
                if relevant_weight < 0.5 + tolerance:
                    # close your eyes and discard
                    self.log("next player has only relevant cards, so I discard")
                    return DiscardAction(card_pos=card_pos)
            
            elif all(card.useful(self.board, self.full_deck, self.discard_pile) for card in self.hands[self.next_player_id()]):
                if relevant_weight < tolerance and useful_weight < 1.0 + tolerance:
                    # discard anyway
                    self.log("next player has only useful cards, so I discard")
                    return DiscardAction(card_pos=card_pos)
        
        
        # try to give hint, using the right hints manager
        hint_action = self.value_hints_manager.get_best_hint()
        
        if hint_action is not None:
            return hint_action
        else:
            # failed to give indirect hint
            # discard card
            self.log("failed to give a hint")
            return DiscardAction(card_pos=self.get_best_discard()[0])


