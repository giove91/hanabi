#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random
from collections import namedtuple
from termcolor import colored

from card import Card, deck
from player import Player
from action import Action


Turn = namedtuple("Turn", "player action")
Statistics = namedtuple("Statistics", "score lives hints num_turns")


class Game:
    CARDS_PER_PLAYER = {2: 5, 3: 5, 4: 4, 5: 4}
    INITIAL_HINTS = 8
    MAX_HINTS = 8
    INITIAL_LIVES = 3
    
    
    def __init__(self, num_players, ai="alphahanabi", wait_key=True, log=True, strategy_log=False, dump_deck_to=None, load_deck_from=None):
        self.num_players = num_players
        self.ai = ai
        
        self.wait_key = wait_key    # press Enter to advance turn
        self.log = log              # log advancement of the game to standard output
        self.strategy_log = strategy_log    # log messages from strategy to standard output
        self.dump_deck_to = dump_deck_to    # if not None, dump the initial deck to the given file
        self.load_deck_from = load_deck_from    # if not None, load the initial deck from the given file
        
        # compute number of cards per player
        self.k = self.CARDS_PER_PLAYER[num_players]
    
    
    def setup(self):
        if self.load_deck_from is None:
            # construct deck
            self.deck = deck()
            
            # shuffle deck
            random.shuffle(self.deck)
        
        else:
            # use given deck
            self.load_deck(self.load_deck_from)
            assert set(self.deck) == set(deck())
        
        if self.dump_deck_to is not None:
            # dump initial deck to file
            self.dump_deck(self.dump_deck_to)
        
        # initialize players, with initial hand of cards
        self.players = [Player(
                id = i,
                game = self,
                hand = [self.draw_card_from_deck() for i in xrange(self.k)],
                ai = self.ai,
                strategy_log = self.strategy_log
            ) for i in xrange(self.num_players)]
        
        # set number of hints and lives
        self.hints = self.INITIAL_HINTS
        self.lives = self.INITIAL_LIVES
        
        # turn log
        self.turns = []
        
        # construct board (cards in play), indicating the last played number for each color
        self.board = {color: 0 for color in Card.COLORS}
        
        # construct discard pile (includes cards on the board)
        self.discard_pile = []
        
        # set last round variable
        self.last_round = False
        self.last_player = None
        self.last_turn = None
        
        # call players' initializations
        for player in self.players:
            player.initialize()
    
    
    def get_current_turn(self):
        return len(self.turns)
    
    def get_current_score(self):
        return sum(self.board.itervalues())
    
    def draw_card_from_deck(self, player=None):
        if len(self.deck) == 1:
            assert not self.last_round
            # set end game condition
            self.last_round = True
            self.last_player = player
            self.last_turn = self.get_current_turn() + self.num_players
        
        if len(self.deck) > 0:
            return self.deck.pop()
        else:
            return None
    
    
    def increment_hints(self):
        self.hints += 1
        self.hints = min(self.hints, self.MAX_HINTS)
    
    def decrement_hints(self):
        if self.hints == 0:
            raise Exception("No hints available")
        self.hints -= 1
    
    def decrement_lives(self):
        self.lives -= 1
        return self.lives == 0
    
    
    def run_turn(self, player):
        action = player.get_turn_action()
        end_game = self.last_round and self.last_player == player
        
        if action.type == Action.PLAY:
            card = player.hand[action.card_pos]
            assert card is not None
            if card.number == self.board[card.color] + 1:
                # play is successful
                self.board[card.color] += 1
                
                if card.number == 5:
                    # increment hints!
                    self.increment_hints()
            else:
                # play is not successful
                end_game = self.decrement_lives() or end_game
            
            # remove card from hand
            self.discard_pile.append(card)
            player.hand[action.card_pos] = self.draw_card_from_deck(player)
        
        elif action.type == Action.DISCARD:
            card = player.hand[action.card_pos]
            assert card is not None
            
            # increment hints
            self.increment_hints()
            
            # remove card from hand
            self.discard_pile.append(card)
            player.hand[action.card_pos] = self.draw_card_from_deck(player)
        
        elif action.type == Action.HINT:
            # decrement hints
            self.decrement_hints()
            
            # check for correctness
            target = self.players[action.player_id]
            assert player != target
            if action.hint_type == Action.COLOR:
                assert any(card is not None and card.color == action.color for card in target.hand)
            else:
                assert any(card is not None and card.number == action.number for card in target.hand)
                
        
        else:
            raise Exception("Unknown action type.")
        
        return Turn(player, action), end_game
    
    
    def log_turn(self, turn, player):
        action = turn.action
        print "Turn %d (player %d):" % (self.get_current_turn(), player.id),
        if action.type in [Action.PLAY, Action.DISCARD]:
            print action.type, self.discard_pile[-1], "(card %d)," % action.card_pos,
            print "draw %r" % player.hand[action.card_pos]
        
        elif action.type == Action.HINT:
            print action.type,
            print "to player %d," % action.player_id,
            print "cards", action.cards_pos,
            print "are",
            print action.value
        
        print
    
    
    def log_status(self):
        print "Hands:"
        for player in self.players:
            print "    Player %d" % player.id, player.hand
        print "Board:",
        for color in Card.COLORS:
            print colored("%d" % self.board[color], Card.PRINTABLE_COLORS[color]),
        print
        print "Hints: %d    Lives: %d    Deck: %d    Score: %d" % (self.hints, self.lives, len(self.deck), self.get_current_score())
        
        if self.last_round:
            print "This is the last round (player %d plays last on turn %d)" % (self.last_player.id, self.last_turn)
        else:
            print
        
        print


    def log_deck(self):
        print "Deck (last card on top):"
        print self.deck
        print


    def dump_deck(self, filename):
        """
        Dump the initial deck to file.
        """
        print "Dumping initial deck to %s" % filename
        file = open(filename, "w")
        
        # dump deck
        for card in self.deck:
            print >> file, "%d %s %d" % (card.number, card.color, card.id)
    
    
    def load_deck(self, filename):
        """
        Load the initial deck from file.
        """
        print "Loading initial deck from %s" % filename
        file = open(filename, "r")
        
        self.deck = []
        for line in file:
            number, color, id = line.split()
            self.deck.append(Card(id=int(id), color=color, number=int(number)))

    
    def run_game(self):
        end_game = False
        current_player = self.players[0]
        
        if self.log:
            self.log_status()
        
        while not end_game:
            if self.wait_key:
                raw_input()
            
            # do turn
            turn, end_game = self.run_turn(current_player)
            
            # log turn and status
            if self.log:
                self.log_turn(turn, current_player)
                self.log_status()
            
            # inform all players
            for player in self.players:
                player.feed_turn(turn)
            
            # store turn
            self.turns.append(turn)
            
            # change current player
            current_player = current_player.next_player()
        
        return Statistics(
            score = self.get_current_score(),
            lives = self.lives,
            hints = self.hints,
            num_turns = len(self.turns)
        )
        
        




