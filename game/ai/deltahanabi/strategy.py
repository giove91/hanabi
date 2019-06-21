#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random
import itertools
import copy
from collections import Counter, namedtuple

from ...action import Action, PlayAction, DiscardAction, HintAction
from ...card import Card, CardAppearance, get_appearance
from ...deck import DECKS
from ...base_strategy import BaseStrategy


import pprint



class CardInfo:
    
    def __init__(self, possible_cards, last_hinted):
        self.possible_cards = possible_cards
        self.last_hinted = last_hinted
    
    def __repr__(self):
        return repr((self.possible_cards, self.last_hinted))


class PublicKnowledge:
    
    def __init__(self, num_players, k, deck, board, discard_pile):
        self.deck_composition = set(deck)
        self.card_info = {i : [CardInfo(possible_cards = copy.copy(self.deck_composition), last_hinted = -1) for j in xrange(k)] for i in xrange(num_players)}
        self.board = board
        self.virtual_board = copy.copy(board)
        self.deck = deck
        self.discard_pile = discard_pile
        self.num_players = num_players
    
    def update(self, board, discard_pile):
        self.board = board
        self.discard_pile = discard_pile
    
    def reset_knowledge(self, player_id, card_pos):
        self.card_info[player_id][card_pos] = CardInfo(possible_cards = copy.copy(self.deck_composition), last_hinted = -1)
    
    def known_cards(self, players = None):
        if players is None:
            players = range(self.num_players)
        return Counter(next(iter(ci.possible_cards)) for i in players for ci in self.card_info[i] if len(ci.possible_cards) == 1)
    
    def update_possible_cards(self):
        while True:
            all_invisible_cards = set(self.deck - Counter(self.discard_pile) - self.known_cards())
            did_something = False
            for ci in self.card_info.itervalues():
                for c in ci:
                    p = c.possible_cards
                    if len(p) > 1 and not p.issubset(all_invisible_cards):
                        did_something = True
                        p &= all_invisible_cards
            if not did_something:
                break
    
    def update_virtual_board(self):
        self.virtual_board = copy.copy(self.board)
        play_cards_on_board(self.virtual_board, self.known_cards())
        '''
        kc = self.known_cards()
        while True:
            did_something = False
            for c, n in self.virtual_board.iteritems():
                if any(d.matches(c, n + 1) for d in kc):
                    self.virtual_board[c] = n + 1
                    did_something = True
            if not did_something:
                break
        '''
    
class HintManager:
    # COLORS_NUM                0   1   2   3   4   5   6
    PLAYABLE_CARDS_NUM =    [   0,  4,  4,  4,  3,  2,  2]
    TRASH_CARDS_NUM =       [   4,  3,  3,  2,  2,  2,  2]
    
    def __init__(self, public_knowledge):
        self.pk = public_knowledge
    
    def decode_hint_number(self, hinter, hint):
        assert hint.type == Action.HINT
        i = 0
        if hint.hint_type == Action.NUMBER:
            i += 2
        #ACTHUNG this may not work during the last turn
        if hint.cards_pos[0] == 0:
            i += 1
        i += 4 * ((hint.player_id - hinter - 1) % self.pk.num_players)
        return i
    
    def decode_number_for_one_player(self, number, player_id, turn, new_pk):
        cols = sorted(c for c, n in self.pk.virtual_board.iteritems() if n < 5)
        playable_num = self.PLAYABLE_CARDS_NUM[len(cols)]
        trash_num = self.TRASH_CARDS_NUM[len(cols)]
        cards = self.cards_for_hint(player_id)
        my_ci = new_pk.card_info[player_id]
        if len(cards) == 0:
            assert number == 0
            return
        if len(cards) == 1:
            p = sorted(self.pk.card_info[player_id][cards[0]].possible_cards)
            if len(p) <= 16:
                my_ci[cards[0]].possible_cards = {p[number]}
                my_ci[cards[0]].last_hinted = turn
                return
        playable_cards = {CardAppearance(c, self.pk.virtual_board[c] + 1) for c in cols}
        if number < len(cols) * playable_num:
            pos, col_index = divmod(number, len(cols))
            col = cols[col_index]
            for i in range(pos):
                # not playable
                my_ci[cards[i]].possible_cards -= playable_cards
                my_ci[cards[i]].last_hinted = turn
            # playable
            my_ci[cards[pos]].possible_cards = {CardAppearance(col, self.pk.virtual_board[col] + 1)}
            my_ci[cards[pos]].last_hinted = turn
        else:
            trash_cards = self.encode_trash_cards()
            for i, b in zip(cards, format(number - len(cols) * playable_num, '0%db' % min(len(cards), trash_num))):
                my_ci[i].possible_cards -= playable_cards
                if b == '1':
                    # trash
                    my_ci[i].possible_cards &= trash_cards
                else:
                    # not trash
                    my_ci[i].possible_cards -= trash_cards
                my_ci[i].last_hinted = turn
    
    def update_public_knowledge(self, hinter, hint, player_id, hands):
        assert hint.type == Action.HINT
        new_pk = copy.deepcopy(self.pk)
        my_number = self.decode_hint_number(hinter, hint)
        # update other players
        for i in hands.keys():
            if i != hinter:
                number = self.encode_hand(i, hands[i])
                self.decode_number_for_one_player(number, i, hint.turn, new_pk)
                my_number -= number
        # update myself
        if player_id != hinter:
            self.decode_number_for_one_player(my_number % 16, player_id, hint.turn, new_pk)
        else:
            assert (my_number % 16) == 0
        # use actual info from the hint
        for i, ci in enumerate(new_pk.card_info[hint.player_id]):
            ci.possible_cards = {c for c in ci.possible_cards if c.matches(hint.color, hint.number) == (i in hint.cards_pos)}
        # finally
        return new_pk
        
    def give_hint(self, hinter, hands):
        num = sum(self.encode_hand(i, h) for i, h in hands.iteritems()) % 16
        for h in [HintAction(i, color = c) for i in hands.keys() for c in Card.COLORS] + [HintAction(i, number = n) for i in hands.keys() for n in range(1, Card.NUM_NUMBERS + 1)]:
            h.cards_pos = [i for i, c in enumerate(hands[h.player_id]) if c is not None and c.matches(h.color, h.number)]
            if h.cards_pos and self.decode_hint_number(hinter, h) == num:
                #TODO choose the hint according to the increase in knowledge
                return h
        return None
    
    def cards_for_hint(self, player_id):
        return sorted([i for i, ci in enumerate(self.pk.card_info[player_id]) if len(ci.possible_cards) >= 2 and any(c.useful(self.pk.virtual_board, self.pk.deck, self.pk.discard_pile) for c in ci.possible_cards)], key = lambda x: self.pk.card_info[player_id][x].last_hinted)
    
    def encode_trash_cards(self):
        remaining_cards = self.pk.deck - Counter(self.pk.discard_pile)
        high_board = play_cards_on_board(copy.copy(self.pk.board), remaining_cards)
        def get_val(c):
            if not c.useful(self.pk.virtual_board, self.pk.deck, self.pk.discard_pile):
                return (0, 0)
            return (1 + int(remaining_cards[c] == 1), high_board[c.color] - c.number)
        invisible_cards = sorted(((c, get_val(c)) for c in (remaining_cards - self.pk.known_cards()).elements()), key = lambda x : x[1])
        if len(invisible_cards) == 0:
            return set()
        i = len(invisible_cards) // 2
        while i + 1 < len(invisible_cards) and invisible_cards[i][1] == invisible_cards[i + 1][1]:
            i += 1
        return {invisible_cards[j][0] for j in range(i + 1)}
    
    def encode_hand(self, player_id, hand):
        cols = sorted(c for c, n in self.pk.virtual_board.iteritems() if n < 5)
        playable_num = self.PLAYABLE_CARDS_NUM[len(cols)]
        trash_num = self.TRASH_CARDS_NUM[len(cols)]
        cards = self.cards_for_hint(player_id)
        if len(cards) == 0:
            return 0
        if len(cards) == 1:
            p = sorted(self.pk.card_info[player_id][cards[0]].possible_cards)
            if len(p) <= 16:
                return p.index(hand[cards[0]])
        playable_cards = [(i, c) for i, c in enumerate(cards[: playable_num]) if hand[c].playable(self.pk.virtual_board)]
        trash_cards = self.encode_trash_cards()
        if len(playable_cards) == 0:
            return len(cols) * playable_num + int(''.join('1' if hand[c] in trash_cards else '0' for c in cards[: trash_num]), base = 2)
        else:
            i, c = playable_cards[0]
            return len(cols) * i + cols.index(hand[c].color)


class Strategy(BaseStrategy):

    def initialize(self, id, num_players, k, board, deck_type, my_hand, hands, discard_pile, deck_size):
        self.id = id
        self.num_players = num_players
        self.other_players = [i for i in range(0, num_players) if i != id]
        self.k = k
        self.board = board
        self.deck = Counter(get_appearance(DECKS[deck_type]()))
        self.my_hand = my_hand
        self.hands = hands
        self.discard_pile = discard_pile
        self.deck_size = deck_size
        self.pk = PublicKnowledge(num_players, k, self.deck, board, discard_pile)
    
    def feed_turn(self, player_id, action):
        if action.type == Action.HINT:
            hm = HintManager(self.pk)
            self.pk = hm.update_public_knowledge(player_id, action, self.id, self.hands)
        else:
            self.pk.reset_knowledge(player_id, action.card_pos)
            for i, h in self.hands.iteritems():
                for j, c in enumerate(h):
                    if c is None:
                        self.pk.card_info[i][j].possible_cards = set()
            for j, c in enumerate(self.my_hand):
                if c is None:
                    self.pk.card_info[self.id][j].possible_cards = set()
        if self.id == 0:
            #pprint.pprint(self.pk.card_info, width = 1)
            self.log(self.pk.virtual_board)
        self.pk.update(self.board, self.discard_pile)
        self.pk.update_possible_cards()
        self.pk.update_virtual_board()
    
    def get_best_discard(self, pi):
        w = {1 : 25, 2 : 4, 3 : 0}
        z = [1, 4, 2]
        remaining = self.deck - Counter(self.discard_pile)
        my_known = Counter(next(iter(ci.possible_cards)) for ci in pi if len(ci.possible_cards) == 1)
        known = my_known + self.pk.known_cards(self.hands.keys())
        visible = my_known + Counter(itertools.chain(*self.hands.itervalues()))
        trash_value = [0] * self.k
        for i, ci in enumerate(pi):
            pc = ci.possible_cards
            if self.my_hand[i] is None:
                trash_value[i] = float('inf')
                continue
            if all(not c.useful(self.board, self.deck, self.discard_pile) for c in pc):
                trash_value[i] = -float('inf')
                continue
            # beware: complicated stuff ahead
            is_known = int(len(pc) == 1)
            for c in pc:
                if not c.useful(self.board, self.deck, self.discard_pile):
                    continue
                score_loss = [0, 0, 0]
                for j, p, q in [(0, remaining, remaining - Counter({c : remaining[c]})), (1, known + Counter([c] * (1 - is_known)), known - Counter([c] * is_known)), (2, visible + Counter([c] * (1 - is_known)), visible - Counter([c] * is_known))]:
                    score_loss[j] = score_board(play_cards_on_board(copy.copy(self.board), p)) - score_board(play_cards_on_board(copy.copy(self.board), q))
                trash_value[i] += w[remaining[c]] + z[0] * score_loss[0] + z[1] * score_loss[1] + z[2] * score_loss[2]
            trash_value[i] /= len(pc)
        self.log(pi)
        self.log(trash_value)
        return min(enumerate(trash_value), key = lambda x: x[1])
    
    def get_turn_action(self):
        pi = self.private_info()
        playable_cards = [i for i, ci in enumerate(pi) if len(ci.possible_cards) > 0 and all(c.playable(self.board) for c in ci.possible_cards)]
        if playable_cards:
            return PlayAction(card_pos = playable_cards[0])
        best_discard, trash_value = self.get_best_discard(pi)
        #self.log('Best discard is {} with value {}'.format(best_discard, trash_value))
        if self.hints == 0:
            return DiscardAction(card_pos = best_discard)
        hm = HintManager(self.pk)
        h = hm.give_hint(self.id, self.hands)
        if h is None:
            self.log('Could not give hint')
            #raise Exception()
            return DiscardAction(card_pos = best_discard)
        else:
            self.log(hm.encode_trash_cards())
            return h
    
    def next_player_id(self, n = 1):
        return (self.id + n) % self.num_players
    
    def private_info(self):
        pi = copy.deepcopy(self.pk.card_info[self.id])
        while True:
            all_invisible_cards = set(self.deck - Counter(self.discard_pile) - Counter(itertools.chain(*self.hands.itervalues())) - Counter({next(iter(ci.possible_cards)) for ci in pi if len(ci.possible_cards) == 1}))
            did_something = False
            for c in pi:
                p = c.possible_cards
                if len(p) > 1 and not p.issubset(all_invisible_cards):
                    did_something = True
                    p &= all_invisible_cards
            if not did_something:
                break
        return pi


def play_cards_on_board(board, cards):
    for c in board:
        while board[c] < 5 and CardAppearance(c, board[c] + 1) in cards:
            board[c] += 1
    return board

def score_board(board):
    return sum(board.itervalues())
