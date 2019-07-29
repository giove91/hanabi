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



LOG2 = [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4]


def get_next_plays(num_players, player_id, hands, board, turn, last_turn, draw_card = None, max_wait = 0):
    w = {1 : 4, 2 : 3, 3 : 2, 4 : 1, 5 : 5}
    def simulate(player_id, hands, board, turn, draw_card, max_wait):
        if (last_turn is not None and turn > last_turn) or not any (c.playable(board) for h in hands.itervalues() for c in h if c is not None):
            return [], (score_board(board), -turn)
        best_card_pos = None
        best_plays = []
        best_score = (-1, 0)
        for i, c in enumerate(hands[player_id]):
            if c is not None and c.playable(board):
                board[c.color] += 1
                hands[player_id][i] = draw_card
                plays, score = simulate((player_id + 1) % num_players, hands, board, turn + 1, None, max(max_wait - 1, 0))
                board[c.color] -= 1
                hands[player_id][i] = c
                if score > best_score or (score == best_score and best_card_pos is not None and w[c.number] > w[hands[player_id][best_card_pos].number]):
                    best_score = score
                    best_plays = plays
                    best_card_pos = i
        if best_card_pos is None or max_wait > 0:
            plays, score = simulate((player_id + 1) % num_players, hands, board, turn + 1, draw_card, max(max_wait - 1, 0))
            if score[0] >= best_score[0]:
                best_score = score
                best_plays = plays
                best_card_pos = None # FIX: he could DISCARD, but he should HINT to wait
        best_plays.append((player_id, best_card_pos))
        return best_plays, best_score
    best_plays, _ = simulate(player_id, hands, board, turn, draw_card, max_wait)
    for p, i in reversed(best_plays):
        yield p, i
        player_id = p
    while True:
        player_id += 1
        yield player_id, None


def get_boards_after_next_plays(num_players, player_id, hands, board, turn, last_turn, draw_card = None):
    for p, i in get_next_plays(num_players, player_id, hands, board, turn, last_turn, draw_card):
        if i is not None:
            board = copy.copy(board)
            board[hands[p][i].color] += 1
        yield board


def update_possible_cards(cards, compute_invisible_cards):
    did_something = True
    while did_something:
        all_invisible_cards = compute_invisible_cards()
        did_something = False
        for c in cards:
            if len(c) > 1 and not c.issubset(all_invisible_cards):
                c &= all_invisible_cards
                did_something = True


class PublicKnowledge:
    
    def __init__(self, ai):
        self.ai = ai
        self.possible_cards = {i : [copy.copy(ai.deck_composition) for _ in xrange(ai.k)] for i in xrange(ai.num_players)}
        self.last_hinted = {i : [-1 for _ in xrange(ai.k)] for i in xrange(ai.num_players)}
    
    def clone(self):
        new_pk = PublicKnowledge(self.ai)
        new_pk.possible_cards = copy.deepcopy(self.possible_cards)
        new_pk.last_hinted = copy.deepcopy(self.last_hinted)
        return new_pk
    
    def reset_knowledge(self, player_id, card_pos):
        self.possible_cards[player_id][card_pos] = copy.copy(self.ai.deck_composition)
        self.last_hinted[player_id][card_pos] = -1
    
    def known_cards(self, players = None):
        if players is None:
            players = range(self.ai.num_players)
        return Counter(next(iter(j)) for p in players for j in self.possible_cards[p] if len(j) == 1)
    
    def update_possible_cards(self):
        update_possible_cards([j for p in self.possible_cards.itervalues() for j in p], lambda: set(self.ai.deck - Counter(self.ai.discard_pile) - self.known_cards()))
    
    def update_matching_hint(self, hint):
        for i in xrange(self.ai.k):
            self.possible_cards[hint.player_id][i] = {c for c in self.possible_cards[hint.player_id][i] if c.matches(color = hint.color, number = hint.number) == (i in hint.cards_pos)}
    
    def get_boards(self, player_id):
        return get_boards_after_next_plays(
            self.ai.num_players,
            player_id,
            {i : [next(iter(j)) if len(j) == 1 else None for j in p] for i, p in self.possible_cards.iteritems()},
            self.ai.board,
            self.ai.turn,
            self.ai.last_turn
        )
    
    def cards_for_hint(self, player_id, useful):
        return sorted((i for i, p in enumerate(self.possible_cards[player_id]) if len(p) > 1 and len(p & useful) > 0), key = lambda x: self.last_hinted[player_id][x])
    
    def colors_for_hint(self, player_id, cards, useful, board):
        s = score_board(self.ai.board)
        if s <= 6:
            num = 15
        elif s <= 12:
            num = 14
        else:
            num = 12
        return sorted(((i, c.color, c.number) for i in cards for c in useful & self.possible_cards[player_id][i]), key = lambda x: x[2] - board[x[1]])[: num]
    
    def encode_hand(self, player_id, board):
        hand = self.ai.hands[player_id]
        useful = useful_cards(board, self.ai.deck, self.ai.discard_pile)
        cards = self.cards_for_hint(player_id, useful)
        if len(cards) == 0:
            return 0
        l = self.colors_for_hint(player_id, cards, useful, board)
        for j, (i, c, n) in enumerate(l):
            if hand[i].matches(color = c, number = n):
                return j
        trash_num = LOG2[16 - len(l)]
        return len(l) + int('0' + ''.join('0' if hand[i] in useful else '1' for i in cards[: trash_num]), base = 2)
    
    def decode_number_from_hint(self, hinter, hint):
        i = 0
        if hint.hint_type == Action.NUMBER:
            i += 2
        if hint.cards_pos[0] == 0:
            i += 1
        i += 4 * ((hint.player_id - hinter - 1) % self.ai.num_players)
        return i
        
    def decode_hand(self, player_id, board, number, new_pk):
        useful = useful_cards(board, self.ai.deck, self.ai.discard_pile)
        cards = self.cards_for_hint(player_id, useful)
        if len(cards) == 0:
            assert number == 0
            return
        l = self.colors_for_hint(player_id, cards, useful, board)
        for j, (i, c, n) in enumerate(l):
            if number == j:
                new_pk.possible_cards[player_id][i] = {CardAppearance(color = c, number = n)}
                new_pk.last_hinted[player_id][i] = self.ai.turn
                return
            else:
                new_pk.possible_cards[player_id][i].discard(CardAppearance(color = c, number = n))
                new_pk.last_hinted[player_id][i] = self.ai.turn
        trash_num = LOG2[16 - len(l)]
        if trash_num > 0:
            for i, b in zip(cards, format(number - len(l), '0%db' % min(len(cards), trash_num))):
                if b == '1':
                    new_pk.possible_cards[player_id][i] -= useful
                else:
                    new_pk.possible_cards[player_id][i] &= useful
                new_pk.last_hinted[player_id][i] = self.ai.turn
    
    def update_with_hint(self, hinter, hint, boards):
        new_pk = self.clone()
        my_number = self.decode_number_from_hint(hinter, hint)
        my_board = None
        for i, b in zip([1, 2, 3, 4], boards):
            p = (hinter + i) % self.ai.num_players
            if p != self.ai.id:
                number = self.encode_hand(p, b)
                self.decode_hand(p, b, number, new_pk)
                my_number -= number
            else:
                my_board = b
        if self.ai.id == hinter:
            assert my_number % 16 == 0
        else:
            self.decode_hand(self.ai.id, my_board, my_number % 16, new_pk)
        new_pk.update_matching_hint(hint)
        new_pk.update_possible_cards()
        self.possible_cards = new_pk.possible_cards
        self.last_hinted = new_pk.last_hinted


class Strategy(BaseStrategy):

    def initialize(self, id, num_players, k, board, deck_type, my_hand, hands, discard_pile, deck_size):
        self.id = id
        self.num_players = num_players
        self.k = k
        self.board = board
        self.deck_type = deck_type
        self.deck = Counter(get_appearance(DECKS[deck_type]()))
        self.deck_composition = set(self.deck)
        self.my_hand = my_hand
        self.hands = hands
        self.discard_pile = discard_pile
        self.deck_size = deck_size
        self.pk = PublicKnowledge(self)
    
    def feed_turn(self, player_id, action):
        if action.type == Action.HINT:
            self.pk.update_with_hint(player_id, action, self.pk.get_boards((player_id + 1) % self.num_players))
        else:
            self.pk.reset_knowledge(player_id, action.card_pos)
            if (player_id == self.id and self.my_hand[action.card_pos] is None) or (player_id != self.id and self.hands[player_id][action.card_pos] is None):
                self.pk.possible_cards[player_id][action.card_pos] = set()
            self.pk.update_possible_cards()
    
    def get_turn_action(self):
        if self.verbose:
            for i in range(self.num_players):
                print [next(iter(c)) if len(c) == 1 else 'T' if len(c & useful_cards(self.board, self.deck, self.discard_pile)) == 0 else '?%d?' % len(c) for c in self.pk.possible_cards[i]]
        pi = self.private_info()
        best_play = self.get_best_play(pi)
        if best_play is not None:
            return PlayAction(best_play)
        best_discard, discard_value = self.get_best_discard(pi)
        if self.hints == 0:
            return DiscardAction(best_discard)
        boards1, boards2 = itertools.tee(self.pk.get_boards(self.next_player_id()), 2)
        h = self.choose_hint(boards1)
        if h is None:
            self.log('Could not give hint')
            return DiscardAction(best_discard)
        if self.deck_size > 1:
            new_pk = self.pk.clone()
            new_pk.update_with_hint(self.id, h, boards2)
            hints = self.hints - 1
            old_b = self.board
            p = self.id
            for b in itertools.islice(new_pk.get_boards(self.next_player_id()), self.num_players - 1):
                p = (p + 1) % self.num_players
                if b != old_b is not None:
                    if any(old_b[col] < 5 for col in b if b[col] == 5):
                        hints += 1
                    old_b = b
                    continue
                u = useful_cards(b, self.deck, self.discard_pile)
                if any(len(j & u) == 0 for j in new_pk.possible_cards[p]):
                    hints += 1
                    continue
                ai = Strategy()
                ai.initialize(p, self.num_players, self.k, self.board, self.deck_type, self.hands[p], copy.copy(self.hands), self.discard_pile, self.deck_size)
                ai.pk = new_pk
                del ai.hands[p]
                ai.hands[self.id] = [next(iter(j)) if len(j) == 1 else None for j in pi]
                _, other_discard_value = ai.get_best_discard(ai.private_info())
                if other_discard_value < discard_value + ((p - self.id) % self.num_players):
                    hints += 1
                else:
                    if hints == 0:
                        self.log('Discard with value {} to prevent {} from discarding with value ~{}'.format(discard_value, p, other_discard_value))
                        return DiscardAction(best_discard)
                    hints -= 1
        u = useful_cards(self.board, self.deck, self.discard_pile)
        if self.deck_size > 1 and self.hints < 4 and discard_value <= 0 and all(c not in u or len(p) == 1 for i, h in self.hands.iteritems() for c, p in zip(h, self.pk.possible_cards[i])):
            self.log('Hinting would be useless')
            return DiscardAction(best_discard)
        return h
    
    def choose_hint(self, boards):
        number = sum(self.pk.encode_hand(self.next_player_id(i), b) for i, b in zip([1, 2, 3, 4], boards)) % 16
        for h in [HintAction(i, color = c) for i in self.hands.keys() for c in Card.COLORS] + [HintAction(i, number = n) for i in self.hands.keys() for n in range(1, Card.NUM_NUMBERS + 1)]:
            h.cards_pos = [i for i, c in enumerate(self.hands[h.player_id]) if c is not None and c.matches(h.color, h.number)]
            if h.cards_pos and self.pk.decode_number_from_hint(self.id, h) == number:
                #TODO choose the hint according to the increase in knowledge
                return h
        return None
    
    def get_best_play(self, pi):
        if self.deck_size <= 1:
            hands = copy.deepcopy(self.hands)
        else:
            hands = {i : [next(iter(j)) if len(j) == 1 else None for j in self.pk.possible_cards[i]] for i in self.hands.iterkeys()}
        hands[self.id] = [next(iter(j)) if len(j) == 1 else None for j in pi]
        my_playable = [i for i, c in enumerate(hands[self.id]) if c is not None and c.playable(self.board)]
        if len(my_playable) == 0:
            return None
        if self.deck_size == 1:
            missing_useful_cards = set(self.deck - Counter(self.discard_pile) - Counter(itertools.chain(*self.hands.itervalues())) - Counter({next(iter(j)) for j in pi if len(j) == 1})) & useful_cards(self.board, self.deck, self.discard_pile)
            if len(missing_useful_cards) > 1 or any(len(j) > 1 and len(j & missing_useful_cards) > 0 for j in pi):
                self.log('Could not determine last card')
            else:
                missing_card = next(iter(missing_useful_cards)) if missing_useful_cards else None
                return next(get_next_plays(self.num_players, self.id, hands, self.board, self.turn, self.last_turn, missing_card, min(self.num_players - 1, self.hints)))[1]
        return next(get_next_plays(self.num_players, self.id, hands, self.board, self.turn, self.last_turn))[1]
    
    def get_best_discard(self, pi):
        w = {1 : 25, 2 : 4, 3 : 0}
        z = [1, 3, 3]
        remaining = self.deck - Counter(self.discard_pile)
        my_known = Counter(next(iter(j)) for j in pi if len(j) == 1)
        known = my_known + self.pk.known_cards(self.hands.keys())
        visible = my_known + Counter(itertools.chain(*self.hands.itervalues()))
        useful = useful_cards(self.board, self.deck, self.discard_pile)
        trash_value = [0.0] * self.k
        for i, pc in enumerate(pi):
            if self.my_hand[i] is None:
                trash_value[i] = float('inf')
                continue
            if len(pc & useful) == 0:
                trash_value[i] = -float('inf')
                continue
            is_known = int(len(pc) == 1)
            for c in pc:
                if c not in useful or my_known[c] > is_known:
                    continue
                board = copy.copy(self.board)
                board[c.color] = c.number - 1
                trash_value[i] += w[remaining[c]]
                for j, p, q in [(0, remaining, remaining - Counter({c : remaining[c]})), (1, known + Counter([c] * (1 - is_known)), known - Counter([c] * is_known)), (2, visible + Counter([c] * (1 - is_known)), visible - Counter([c] * is_known))]:
                    trash_value[i] += z[j] * (score_board(board, p) - score_board(board, q))
            trash_value[i] /= len(pc)
        self.log(trash_value)
        return min(enumerate(trash_value), key = lambda x: x[1])

    def private_info(self):
        pi = copy.deepcopy(self.pk.possible_cards[self.id])
        update_possible_cards(pi, lambda: set(self.deck - Counter(self.discard_pile) - Counter(itertools.chain(*self.hands.itervalues())) - Counter({next(iter(j)) for j in pi if len(j) == 1})))
        return pi
    
    def next_player_id(self, n = 1):
        return (self.id + n) % self.num_players

def score_board(board, cards_to_play = None):
    if cards_to_play is not None:
        board = copy.copy(board)
        for col in board:
            while board[col] < 5 and CardAppearance(col, board[col] + 1) in cards_to_play:
                board[col] += 1
    return sum(board.itervalues())

def useful_cards(board, deck, discard_pile):
    return {c for c in set(deck) if c.useful(board, deck, discard_pile)}
