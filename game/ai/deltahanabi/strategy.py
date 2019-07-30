#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random
import itertools
import copy
from collections import Counter, defaultdict

from ...action import Action, PlayAction, DiscardAction, HintAction
from ...card import Card, CardAppearance, get_appearance
from ...deck import DECKS
from ...base_strategy import BaseStrategy



LOG2 = [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4]


def get_next_plays(num_players, player_id, hands, board, turn, last_turn, full_deck, discard_pile, deck = None, hints = 0, all_plays = False):
    w = {1 : 4, 2 : 3, 3 : 2, 4 : 1, 5 : 5}
    def simulate(player_id, hands, board, turn, last_turn, deck, hints, all_plays = False):
        if (last_turn is not None and turn > last_turn) or not any (c.playable(board) for h in hands.itervalues() for c in h if c is not None):
            if all_plays:
                return {None : ([], (score_board(board), -turn))}
            return [], (score_board(board), -turn)
        plays = {None : ([], (-1, 0))}
        if hints > 0 and deck is not None:
            plays['hint'] = simulate((player_id + 1) % num_players, hands, board, turn + 1, last_turn, deck, hints - 1)
        draw_card = None if deck is None or len(deck) == 0  else deck[-1]
        if deck is not None and len(deck) > 0:
            if len(deck) == 1:
                last_turn = turn + num_players
            deck = deck[: -1]
        for i, c in enumerate(hands[player_id]):
            if c is not None and c.playable(board):
                board[c.color] += 1
                hands[player_id][i] = draw_card
                plays[i] = simulate((player_id + 1) % num_players, hands, board, turn + 1, last_turn, deck, hints + int(c.number == 5))
                board[c.color] -= 1
                hands[player_id][i] = c
        if deck is None and len(plays) == 1:
            plays[None] = simulate((player_id + 1) % num_players, hands, board, turn + 1, last_turn, deck, hints)
        if deck is not None:
            u = useful_cards(board, full_deck, discard_pile)
            if all(c in u for c in hands[player_id]):
                discardable = enumerate(hands[player_id])
            else:
                discardable = [next((i, c) for i, c in enumerate(hands[player_id]) if c not in u)]
            for i, c in discardable:
                hands[player_id][i] = draw_card
                plays['discard%d' % i] = simulate((player_id + 1) % num_players, hands, board, turn + 1, last_turn, deck, hints + 1)
                hands[player_id][i] = c
        if all_plays:
            return plays
        play = max(plays, key = lambda p: (plays[p][1], w[hands[player_id][p].number] if isinstance(p, (int, long)) else None))
        plays[play][0].append((player_id, play))
        return plays[play]
    if all_plays:
        yield {p : s[1][0] for p, s in simulate(player_id, hands, board, turn, last_turn, deck, hints, True).iteritems() if s >= 0}
    else:
        best_plays, best_score = simulate(player_id, hands, board, turn, last_turn, deck, hints)
        for p, i in reversed(best_plays):
            yield p, i, best_score[0]
            player_id = p
        while True:
            player_id += 1
            yield player_id, None, best_score[0]


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
        hands = {i : [next(iter(j)) if len(j) == 1 else None for j in p] for i, p in self.possible_cards.iteritems()}
        plays = get_next_plays(
            self.ai.num_players,
            player_id,
            hands,
            self.ai.board,
            self.ai.turn,
            self.ai.last_turn,
            self.ai.deck,
            self.ai.discard_pile
        )
        board = self.ai.board
        for p, i, _ in plays:
            if isinstance(i, (int, long)):
                board = copy.copy(board)
                assert hands[p][i].playable(board)
                board[hands[p][i].color] += 1
            yield board
    
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
        return sorted(((i, c.color, c.number) for i in cards for c in useful & self.possible_cards[player_id][i]), key = lambda x: (x[2] - board[x[1]], self.last_hinted[player_id][x[0]], x[0]))[: num]
    
    def safe_discards(self, player_id, discard_pile):
        useful = useful_cards(self.ai.board, self.ai.deck, discard_pile)
        known = self.known_cards([player_id])
        return [i for i, p in enumerate(self.possible_cards[player_id]) if len(p) > 0 and (len(p & useful) == 0 or (len(p) == 1 and known[next(iter(p))] > 1))]
    
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
    
    def encode_hand2(self, player_id, board, discard_pile, mod):
        hand = self.ai.hands[player_id]
        useful = useful_cards(board, self.ai.deck, discard_pile)
        cards = [i for i, p in enumerate(self.possible_cards[player_id]) if len(p) > 1 and len(p & useful) > 0]
        if len(cards) == 0:
            return 0
        i = max(cards, key = lambda j: len(self.possible_cards[player_id][j] & useful))
        if self.possible_cards[player_id][i].issubset(useful):
            return sorted(self.possible_cards[player_id][i]).index(self.ai.hands[player_id][i]) % mod
        if hand[i] in useful:
            return 1 + (sorted(self.possible_cards[player_id][i] & useful).index(self.ai.hands[player_id][i]) % (mod - 1))
        else:
            return 0
    
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
    
    def decode_hand2(self, player_id, board, discard_pile, mod, number, new_pk):
        useful = useful_cards(board, self.ai.deck, discard_pile)
        cards = [i for i, p in enumerate(self.possible_cards[player_id]) if len(p) > 1 and len(p & useful) > 0]
        if len(cards) == 0:
            assert number == 0
            return
        i = max(cards, key = lambda j: len(self.possible_cards[player_id][j] & useful))
        if self.possible_cards[player_id][i].issubset(useful):
            p = sorted(new_pk.possible_cards[player_id][i])
            new_pk.possible_cards[player_id][i] = {p[j] for j in xrange(number, len(p), mod)}
        elif number == 0:
            new_pk.possible_cards[player_id][i] -= useful
        else:
            p = sorted(new_pk.possible_cards[player_id][i] & useful)
            new_pk.possible_cards[player_id][i] = {p[j] for j in xrange(number - 1, len(p), mod - 1)}
    
    def update_with_hint(self, hinter, hint, boards):
        new_pk = self.clone()
        my_number = self.decode_number_from_hint(hinter, hint)
        my_board = None
        for i, b in zip(xrange(1, self.ai.num_players), boards):
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
    
    def update_with_discard(self, discarder, action, boards):
        dp = self.ai.discard_pile[: -1]
        d = self.safe_discards(discarder, dp)
        mod = len(d)
        if mod <= 1:
            return
        new_pk = self.clone()
        my_number = d.index(action.card_pos)
        my_board = None
        for i, b in zip(xrange(1, self.ai.num_players), boards):
            p = (discarder + i) % self.ai.num_players
            if p != self.ai.id:
                number = self.encode_hand2(p, b, dp, mod)
                self.decode_hand2(p, b, dp, mod, number, new_pk)
                my_number -= number
            else:
                my_board = b
        if self.ai.id == discarder:
            assert my_number % mod == 0
        else:
            self.decode_hand2(self.ai.id, my_board, dp, mod, my_number % mod, new_pk)
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
            if action.type == Action.DISCARD:
                self.pk.update_with_discard(player_id, action, self.pk.get_boards((player_id + 1) % self.num_players))
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
        if isinstance(best_play, (int, long)):
            return PlayAction(best_play)
        best_discard, discard_value = self.get_best_discard(pi)
        boards1, boards2, boards3 = itertools.tee(self.pk.get_boards(self.next_player_id()), 3)
        safe_discards = self.pk.safe_discards(self.id, self.discard_pile)
        discard_action = DiscardAction(best_discard)
        if len(safe_discards) >= 2:
            discard_action = DiscardAction(self.choose_discard_for_hint(safe_discards, boards3))
        if self.hints == 0 or best_play == 'discard':
            return discard_action
        h = self.choose_hint(boards1)
        if h is None:
            self.log('Could not give hint')
            return discard_action
        if best_play == 'hint':
            return h
        if best_play == 'discard':
            return discard_action
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
                if other_discard_value <= max(0, discard_value + ((p - self.id) % self.num_players)):
                    hints += 1
                else:
                    if hints == 0:
                        self.log('Discard with value {} to prevent {} from discarding with value ~{}'.format(discard_value, p, other_discard_value))
                        return discard_action
                    hints -= 1
        u = useful_cards(self.board, self.deck, self.discard_pile)
        if self.deck_size > 1 and self.hints < 4 and discard_value <= 0 and all(c not in u or len(p) == 1 for i, h in self.hands.iteritems() for c, p in zip(h, self.pk.possible_cards[i])):
            self.log('Hinting would be useless')
            return discard_action
        return h
    
    def choose_hint(self, boards):
        number = 0
        new_pk = self.pk.clone()
        u = {}
        for i, b in zip(map(self.next_player_id, xrange(1, self.num_players)), boards):
            n = self.pk.encode_hand(i, b)
            self.pk.decode_hand(i, b, n, new_pk)
            number += n
            u[i] = useful_cards(b, self.deck, self.discard_pile)
        number %= 16
        best_hint = None
        best_score = (0, -float('inf'))
        for h in [HintAction(i, color = c) for i in self.hands.keys() for c in Card.COLORS] + [HintAction(i, number = n) for i in self.hands.keys() for n in range(1, Card.NUM_NUMBERS + 1)]:
            h.cards_pos = [i for i, c in enumerate(self.hands[h.player_id]) if c is not None and c.matches(h.color, h.number)]
            if h.cards_pos and self.pk.decode_number_from_hint(self.id, h) == number:
                p = [{c for c in q if c.matches(h.color, h.number) == (i in h.cards_pos)} for i, q in enumerate(new_pk.possible_cards[h.player_id])]
                score = (sum(1 for c in p if len(c) == 1 or len(c & u[h.player_id]) == 0), -sum(len(c & u[h.player_id]) for c in p))
                if score > best_score:
                    best_score = score
                    best_hint = h
        return best_hint
    
    def choose_discard_for_hint(self, safe_discards, boards):
        mod = len(safe_discards)
        number = sum(self.pk.encode_hand2(self.next_player_id(i), b, self.discard_pile, mod) for i, b in zip(xrange(1, self.num_players), boards)) % mod
        self.log('Discarding %dth safe card to give extra info' % number)
        return safe_discards[number]
    
    def get_best_play(self, pi):
        if self.deck_size <= 3:
            w = {None : 0, 'discard': 1, 'hint': 2, 0: 3, 1: 3, 2: 3, 3: 3}
            hands = copy.deepcopy(self.hands)
            def best_play_from_plays(plays):
                play = max(plays, key = lambda p: (int(plays[p] * 10), w[p]))
                return play, plays[play]
            def simulate(player_id, hands, board, turn, last_turn, deck, hints):
                s = float(score_board(board))
                plays = defaultdict(lambda: s)
                plays[None] = s
                if (last_turn is not None and turn > last_turn) or not any (c.playable(board) for c in itertools.chain(deck, *hands.itervalues()) if c is not None):
                    return plays
                if hints > 0:
                    plays['hint'] = best_play_from_plays(simulate((player_id + 1) % self.num_players, hands, board, turn + 1, last_turn, deck, hints - 1))[1]
                def simulate0(draw_card, deck, last_turn, mult = 1.0):
                    for i, c in enumerate(hands[player_id]):
                        if c is not None and c.playable(board):
                            board[c.color] += 1
                            hands[player_id][i] = draw_card
                            if i not in plays:
                                plays[i] = 0.0
                            plays[i] += mult * best_play_from_plays(simulate((player_id + 1) % self.num_players, hands, board, turn + 1, last_turn, deck, hints + int(c.number == 5)))[1]
                            board[c.color] -= 1
                            hands[player_id][i] = c
                    u = useful_cards(board, self.deck, self.discard_pile)
                    discardable = next(([(i, c)] for i, c in enumerate(hands[player_id]) if c not in u), [])
                    for i, c in discardable:
                        hands[player_id][i] = draw_card
                        if 'discard' not in plays:
                            plays['discard'] = 0.0
                        plays['discard'] += mult * best_play_from_plays(simulate((player_id + 1) % self.num_players, hands, board, turn + 1, last_turn, deck, hints + 1))[1]
                        hands[player_id][i] = c
                if len(deck) > 0:
                    for draw_card, mult in Counter(deck).iteritems():
                        deck0 = copy.copy(deck)
                        deck0.remove(draw_card)
                        if len(deck) == 1:
                            last_turn = turn + self.num_players
                        simulate0(draw_card, deck0, last_turn, mult / float(len(deck)))
                else:
                    simulate0(None, [], last_turn)
                return plays
            u = useful_cards(self.board, self.deck, self.discard_pile)
            f = lambda x: x if x in u else None
            s = score_board(self.board)
            plays = {}
            count = 0
            for h, mult in Counter(itertools.product(*(map(f, p) for p in pi))).iteritems():
                hands[self.id] = []
                deck = Counter(map(f, (self.deck - Counter(self.discard_pile)).elements())) -  Counter(f(c) for h in self.hands.itervalues() for c in h if c is not None)
                h_counter = Counter(h)
                if any(h_counter[c] > deck[c] for c in h_counter):
                    continue
                hands[self.id] = list(h)
                deck = list((deck - h_counter).elements())
                self.log((h, deck))
                p = simulate(self.id, hands, self.board, self.turn, self.last_turn, deck, self.hints)
                self.log(p)
                for a in p:
                    if a not in plays:
                        plays[a] = s * count
                for a in plays:
                    plays[a] += p[a] * mult
                count += mult
            for p in plays:
                plays[p] /= count
            self.log(plays)
            return best_play_from_plays(plays)[0]
        else:
            hands = {i : [next(iter(j)) if len(j) == 1 else None for j in self.pk.possible_cards[i]] for i in self.hands.iterkeys()}
            hands[self.id] = [next(iter(j)) if len(j) == 1 else None for j in pi]
            if all(c is None or not c.playable(self.board) for c in hands[self.id]):
                return None
            return next(get_next_plays(self.num_players, self.id, hands, self.board, self.turn, self.last_turn, self.deck, self.discard_pile))[1]
    
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
