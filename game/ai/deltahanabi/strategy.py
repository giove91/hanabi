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
        self.k = k
        self.virtual_board = copy.copy(board)
        self.deck = deck
        self.discard_pile = discard_pile
        self.num_players = num_players
        self.update_useful_cards()
    
    def update(self, board, discard_pile):
        self.board = board
        self.discard_pile = discard_pile
        self.update_possible_cards()
        self.update_virtual_board()
        self.update_useful_cards()
    
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
    
    def update_useful_cards(self):
        self.useful = {c for c in self.deck_composition if c.useful(self.board, self.deck, self.discard_pile)}
        self.virtual_useful = {c for c in self.useful if c.useful(self.virtual_board, self.deck, self.discard_pile)}
    
    
    
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
        cards = self.cards_for_hint(player_id)
        playable_num = self.pk.k
        while True:
            cols = sorted(c for c, n in self.pk.board.iteritems() if any(d.color == c and d.number > n for i in cards[: playable_num] for d in self.pk.card_info[player_id][i].possible_cards))
            if self.PLAYABLE_CARDS_NUM[len(cols)] < playable_num:
                playable_num -= 1
            else:
                break
        trash_num = self.TRASH_CARDS_NUM[len(cols)]
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
        if number < len(cols) * playable_num:
            pos, col_index = divmod(number, len(cols))
            col = cols[col_index]
            for i in range(pos):
                # not "playable"
                unplayable = {}
                for c in my_ci[cards[i]].possible_cards:
                    if c in self.pk.useful and (c.color not in unplayable or unplayable[c.color] > c.number):
                        unplayable[c.color] = c.number
                my_ci[cards[i]].possible_cards -= {CardAppearance(c, n) for c, n in unplayable.iteritems()}
                my_ci[cards[i]].last_hinted = turn
            # "playable"
            my_ci[cards[pos]].possible_cards = {CardAppearance(col, min(c.number for c in my_ci[cards[pos]].possible_cards if c.matches(col, None) and c in self.pk.useful))}
            my_ci[cards[pos]].last_hinted = turn
        else:
            for i in cards[: playable_num]:
                unplayable = {}
                for c in my_ci[i].possible_cards:
                    if c in self.pk.useful and (c.color not in unplayable or unplayable[c.color] > c.number):
                        unplayable[c.color] = c.number
                my_ci[i].possible_cards -= {CardAppearance(c, n) for c, n in unplayable.iteritems()}
            trash_cards = self.encode_trash_cards()
            for i, b in zip(cards, format(number - len(cols) * playable_num, '0%db' % min(len(cards), trash_num))):
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
        trash_cards = self.encode_trash_cards()
        #return sorted([i for i, ci in enumerate(self.pk.card_info[player_id]) if len(ci.possible_cards) >= 2 and any(c.useful(self.pk.virtual_board, self.pk.deck, self.pk.discard_pile) for c in ci.possible_cards)], key = lambda x: self.pk.card_info[player_id][x].last_hinted)
        return sorted([i for i, ci in enumerate(self.pk.card_info[player_id]) if len(ci.possible_cards) >= 2 and any(c not in trash_cards for c in ci.possible_cards)], key = lambda x: (not any(c.playable(self.pk.virtual_board) for c in self.pk.card_info[player_id][x].possible_cards), self.pk.card_info[player_id][x].last_hinted))
    
    def encode_trash_cards(self):
        #return set()
        return self.pk.deck_composition - self.pk.useful
        remaining_cards = self.pk.deck - Counter(self.pk.discard_pile)
        high_board = play_cards_on_board(copy.copy(self.pk.board), remaining_cards)
        def get_val(c):
            if c not in self.pk.virtual_useful:
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
        cards = self.cards_for_hint(player_id)
        playable_num = self.pk.k
        while True:
            cols = sorted(c for c, n in self.pk.board.iteritems() if any(d.color == c and d.number > n for i in cards[: playable_num] for d in self.pk.card_info[player_id][i].possible_cards))
            if self.PLAYABLE_CARDS_NUM[len(cols)] < playable_num:
                playable_num -= 1
            else:
                break
        trash_num = self.TRASH_CARDS_NUM[len(cols)]
        if len(cards) == 0:
            return 0
        if len(cards) == 1:
            p = sorted(self.pk.card_info[player_id][cards[0]].possible_cards)
            if len(p) <= 16:
                return p.index(hand[cards[0]])
        for i, j in enumerate(cards[: playable_num]):
            if hand[j].number ==  min([6] + [c.number for c in self.pk.card_info[player_id][j].possible_cards if c.matches(hand[j].color, None) and c in self.pk.useful]):
                return len(cols) * i + cols.index(hand[j].color)
        trash_cards = self.encode_trash_cards()
        return len(cols) * playable_num + int(''.join('1' if hand[c] in trash_cards else '0' for c in cards[: trash_num]), base = 2)


class Strategy(BaseStrategy):

    def initialize(self, id, num_players, k, board, deck_type, my_hand, hands, discard_pile, deck_size):
        self.id = id
        self.num_players = num_players
        self.other_players = [i for i in range(0, num_players) if i != id]
        self.k = k
        self.board = board
        self.deck_type = deck_type
        self.deck = Counter(get_appearance(DECKS[deck_type]()))
        self.my_hand = my_hand
        self.hands = hands
        self.discard_pile = discard_pile
        self.deck_size = deck_size
        self.pk = PublicKnowledge(num_players, k, self.deck, board, discard_pile)
        self.useless_hints = 0
    
    def feed_turn(self, player_id, action):
        if action.type == Action.HINT:
            hm = HintManager(self.pk)
            new_pk = hm.update_public_knowledge(player_id, action, self.id, self.hands)
            new_pk.update(self.board, self.discard_pile)
            #if score_board(self.pk.virtual_board) == score_board(new_pk.virtual_board):
            #    self.useless_hints += 1
            #else:
            #    self.useless_hints = 0
            self.pk = new_pk
        else:
            self.useless_hints = 0
            self.pk.reset_knowledge(player_id, action.card_pos)
            for i, h in self.hands.iteritems():
                for j, c in enumerate(h):
                    if c is None:
                        self.pk.card_info[i][j].possible_cards = set()
            for j, c in enumerate(self.my_hand):
                if c is None:
                    self.pk.card_info[self.id][j].possible_cards = set()
            self.pk.update(self.board, self.discard_pile)
    
    def get_best_discard(self, pi):
        w = {1 : 25, 2 : 4, 3 : 0}
        z = [1, 3, 3]
        remaining = self.deck - Counter(self.discard_pile)
        my_known = Counter(next(iter(ci.possible_cards)) for ci in pi if len(ci.possible_cards) == 1)
        known = my_known + self.pk.known_cards(self.hands.keys())
        visible = my_known + Counter(itertools.chain(*self.hands.itervalues()))
        trash_value = [0.0] * self.k
        for i, ci in enumerate(pi):
            pc = ci.possible_cards
            if self.my_hand[i] is None:
                trash_value[i] = float('inf')
                continue
            if all(c not in self.pk.useful for c in pc):
                trash_value[i] = -float('inf')
                continue
            # beware: complicated stuff ahead
            is_known = int(len(pc) == 1)
            for c in pc:
                if c not in self.pk.useful or my_known[c] > is_known:
                    continue
                board = copy.copy(self.board)
                board[c.color] = c.number - 1
                trash_value[i] += w[remaining[c]]
                for j, p, q in [(0, remaining, remaining - Counter({c : remaining[c]})), (1, known + Counter([c] * (1 - is_known)), known - Counter([c] * is_known)), (2, visible + Counter([c] * (1 - is_known)), visible - Counter([c] * is_known))]:
                    trash_value[i] += z[j] * (score_board(play_cards_on_board(copy.copy(board), p)) - score_board(play_cards_on_board(copy.copy(board), q)))
            trash_value[i] /= len(pc)
        self.log(trash_value)
        return min(enumerate(trash_value), key = lambda x: x[1])
    
    def get_best_play(self, pi):
        w = {1 : 4, 2 : 3, 3 : 2, 4 : 1, 5 : 5}
        def simulate(player_id, hands, board, turn, last_turn, draw_card = None):
            if (last_turn is not None and turn > last_turn) or not any(c.playable(board) for h in hands.itervalues() for c in h if c is not None):
                return None, (score_board(board), -turn)
            best_card_pos = None
            best_score = (-1, 0)
            for i, c in enumerate(hands[player_id]):
                if c is not None and c.playable(board):
                    board[c.color] += 1
                    hands[player_id][i] = draw_card
                    _, score = simulate((player_id + 1) % self.num_players, hands, board, turn + 1, last_turn, None)
                    board[c.color] -= 1
                    hands[player_id][i] = c
                    if score > best_score or (score == best_score and best_card_pos is not None and w[hands[player_id][i].number] > w[hands[player_id][best_card_pos].number]):
                        best_score = score
                        best_card_pos = i
            if best_card_pos is None:
                _, best_score = simulate((player_id + 1) % self.num_players, hands, board, turn + 1, last_turn, draw_card)
            return best_card_pos, best_score
        if self.deck_size <= 1:
            hands = copy.deepcopy(self.hands)
        else:
            hands = {i : [next(iter(ci.possible_cards)) if len(ci.possible_cards) == 1 else None for ci in self.pk.card_info[i]] for i in self.hands.iterkeys()}
        hands[self.id] = [next(iter(ci.possible_cards)) if len(ci.possible_cards) == 1 else None for ci in pi]
        my_playable = [i for i, c in enumerate(hands[self.id]) if c is not None and c.playable(self.board)]
        if len(my_playable) == 0:
            return None, None
        if self.deck_size == 1:
            missing_useful_cards = self.pk.useful - set(itertools.chain(*hands.itervalues()))
            if len(missing_useful_cards) > 1:
                self.log('Cannot determine last card')
            else:
                missing_card = next(iter(missing_useful_cards)) if missing_useful_cards else None
                best_card_pos, my_score = simulate(self.id, hands, copy.copy(self.board), self.turn, self.turn + self.num_players, missing_card)
                self.log('If I play, we score {}'.format(my_score))
                for wait in range(1, min(self.num_players, self.hints + 1)):
                    _, score = simulate(self.next_player_id(wait), hands, copy.copy(self.board), self.turn + wait, self.turn + wait + self.num_players, missing_card)
                    if score[0] >= my_score[0]:
                        self.log('If I wait and let player {} play, we score {} instead of {}'.format(self.next_player_id(wait), score, my_score))
                        return None, score
                return best_card_pos, my_score
        if len(my_playable) == 1:
            return my_playable[0], None
        return simulate(self.id, hands, copy.copy(self.board), self.turn, self.last_turn)
    
    def simulate_best_play(self, first_player, board0, turns):
        best_score = 0
        for comb in itertools.product(range(self.k), repeat = turns):
            board = copy.copy(board0)
            player_id = first_player
            for j in comb:
                d = self.hands[player_id][j]
                if d is not None and d.playable(board):
                    board[d.color] += 1
                player_id = (player_id + 1) % self.num_players
            best_score = max(best_score, score_board(board))
        return best_score
    
    def get_best_play_before_last_turn(self, pi):
        if self.deck_size > 1:
            return None, None
        self.hands[self.id] = [None] * self.k
        for i, ci in enumerate(pi):
            if len(ci.possible_cards) == 1:
                self.hands[self.id][i] = next(iter(ci.possible_cards))
        missing_useful_cards = self.pk.useful - set(itertools.chain(*self.hands.itervalues()))
        if len(missing_useful_cards) > 1:
            del self.hands[self.id]
            return None, None
        missing_card = None
        if len(missing_useful_cards) > 0:
            missing_card = next(iter(missing_useful_cards))
        best_player, best_card_pos, best_score = self.id, None, 0
        for i in range(self.hints):
            player_id = self.next_player_id(i)
            for card_pos in range(self.k):
                c = self.hands[player_id][card_pos]
                if c is None:
                    continue
                if c.playable(self.board):
                    board0 = copy.copy(self.board)
                    board0[c.color] += 1
                    self.hands[player_id][card_pos] = missing_card
                    score = self.simulate_best_play((player_id + 1) % self.num_players, board0, self.num_players)
                    if score > best_score:
                        best_player = player_id
                        best_card_pos = card_pos
                        best_score = score
                    self.hands[player_id][card_pos] = c
            if best_player != self.id:
                break
        del self.hands[self.id]
        if best_player == self.id:
            return best_card_pos, best_score
        else:
            return None, None
    
    def get_best_play_last_turn(self, pi):
        # shamelessly stolen from alphahanabi
        best_card_pos = None
        best_avg_score = float(self.simulate_best_play(self.next_player_id(), self.board, self.last_turn - self.turn))
        for i in range(self.k):
            if self.my_hand[i] is None:
                continue
            pc = pi[i].possible_cards
            avg_score = 0.0
            for c in pc:
                board0 = copy.copy(self.board)
                lives = self.lives
                if c.playable(board0):
                    board0[c.color] += 1
                else:
                    lives -= 1
                if lives >= 1:
                    avg_score += self.simulate_best_play(self.next_player_id(), board0, self.last_turn - self.turn)
                else:
                    avg_score += score_board(board0)
            avg_score /= len(pc)
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_card_pos = i
        return best_card_pos, best_avg_score
        
    
    def get_turn_action(self):
        if self.verbose:
            for i in range(self.num_players):
                print [next(iter(ci.possible_cards)) if len(ci.possible_cards) == 1 else 'T' if len(ci.possible_cards & self.pk.useful) == 0 else '?%d?' % len(ci.possible_cards) for ci in self.pk.card_info[i]]
        pi = self.private_info()
        for i in range(self.k):
            if len(pi[i].possible_cards) == 1 and len(self.pk.card_info[self.id][i].possible_cards) > 1 and next(iter(pi[i].possible_cards)) in self.pk.useful:
                self.log('I know card {} but they don\'t'.format(i))
        # play
        best_play, play_value = self.get_best_play(pi)
        if best_play is not None:
            return PlayAction(card_pos = best_play)
        '''
        best_play, play_value = self.get_best_play_before_last_turn(pi)
        if best_play is not None:
            return PlayAction(card_pos = best_play)
        if self.last_turn is None:
            best_play, play_value = self.get_best_play(pi)
            self.log((best_play, play_value))
            if best_play is not None:
                return PlayAction(card_pos = best_play)
        else:
            best_play, play_value = self.get_best_play_last_turn(pi)
            if best_play is not None:
                return PlayAction(card_pos = best_play)
        '''
        # assume I give a hint: what happens?
        hm = HintManager(self.pk)
        h = hm.give_hint(self.id, self.hands)
        if h is None:
            self.log('Could not give hint')
        else:
            h.cards_pos = [i for i, c in enumerate(self.hands[h.player_id]) if c is not None and c.matches(h.color, h.number)]
            h.turn = self.turn
            new_pk = hm.update_public_knowledge(self.id, h, self.id, self.hands)
            new_pk.update(self.board, self.discard_pile)
        # discard
        best_discard, trash_value = self.get_best_discard(pi)
        if self.hints == 0:
            return DiscardAction(card_pos = best_discard)
        if self.deck_size > 1 and h is not None:
            do_discard = False
            needed_hints = 1 - self.hints
            new_board = copy.copy(self.board)
            new_discard_pile = copy.copy(self.discard_pile)
            for i in map(self.next_player_id, range(1, self.num_players)):
                ci = new_pk.card_info[i]
                playable = [p.possible_cards for p in ci if len(p.possible_cards) > 0 and all(c.playable(new_board) for c in p.possible_cards)]
                if len(playable) > 0:
                    # he can play
                    if len({c for p in playable for c in p}) == 1:
                        c = next(iter(playable[0]))
                        new_board[c.color] += 1
                        if c.number == 5:
                            needed_hints -= 1
                        new_discard_pile.append(c)
                        new_pk.update(new_board, new_discard_pile)
                    continue
                if  any(len(p.possible_cards & new_pk.useful) == 0 for p in ci):
                    # he can discard a useless card
                    needed_hints -= 1
                elif all(len(p.possible_cards & new_pk.useful) > 0 for p in ci):
                    # predict what he will discard
                    other_ai = Strategy()
                    other_ai.initialize(i, self.num_players, self.k, self.board, self.deck_type, self.hands[i], copy.copy(self.hands), self.discard_pile, self.deck_size)
                    other_ai.pk = self.pk
                    del other_ai.hands[i]
                    other_ai.hands[self.id] = [None] * self.k
                    for j, di in enumerate(pi):
                        if len(di.possible_cards) == 1:
                            other_ai.hands[self.id][j] = next(iter(di.possible_cards))
                    _, other_trash_value = other_ai.get_best_discard(other_ai.private_info())
                    if other_trash_value >= max(5, trash_value + 3):
                        needed_hints += 1
                        if needed_hints > 0:
                            do_discard = True
                            break
            if do_discard:
                self.log('Discard with value {} to prevent someone else from discarding'.format(trash_value))
                return DiscardAction(card_pos = best_discard)
        if self.deck_size > 1 and self.hints < 4 and self.deck_size >= 10 and trash_value <= 0 and all(c not in self.pk.useful or len(ci.possible_cards) == 1 for p, h in self.hands.iteritems() for c, ci in itertools.izip(h, self.pk.card_info[p])):
            self.log('Hinting would be useless')
            h = None
        # hint
        if h is None:
            return DiscardAction(card_pos = best_discard)
        else:
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
