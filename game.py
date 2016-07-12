import random
from collections import namedtuple
from termcolor import colored

from card import Card, deck
from player import Player
from action import Action


Turn = namedtuple("Turn", "player action")



class Game:
    CARDS_PER_PLAYER = {2: 5, 3: 5, 4: 4, 5: 4}
    INITIAL_HINTS = 8
    MAX_HINTS = 8
    INITIAL_LIVES = 3
    
    
    def __init__(self, num_players):
        self.num_players = num_players
        
        # compute number of cards per player
        self.k = self.CARDS_PER_PLAYER[num_players]
    
    
    
    def setup(self):
        # construct deck
        self.deck = deck()
        
        # shuffle deck
        random.shuffle(self.deck)
        
        # initialize players, with initial hand of cards
        self.players = [Player(
                id = i,
                game = self,
                hand = [self.draw_card_from_deck() for i in xrange(self.k)]
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
    
    
    def draw_card_from_deck(self, player=None):
        if len(self.deck) > 0:
            return self.deck.pop()
        else:
            if not self.last_round:
                # set end game condition
                self.last_round = True
                self.last_player = player
                self.last_turn = self.get_current_turn() + self.num_players
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
            else:
                # play is not successful
                end_game = end_game or self.decrement_lives()
            
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
            if action.color is not None:
                assert any(card.color == action.color for card in target.hand)
            elif action.number is not None:
                assert any(card.number == action.number for card in target.hand)
                
        
        else:
            raise Exception("Unknown action type.")
        
        return Turn(player, action), end_game
    
    
    def log_turn(self, turn, player):
        action = turn.action
        print "Turn %d (player %d):" % (self.get_current_turn(), player.id),
        if action.type in [Action.PLAY, Action.DISCARD]:
            print action.type, self.discard_pile[-1], "(card %d)" % action.card_pos
        
        elif action.type == Action.HINT:
            print action.type,
            print "to player %d," % action.player_id,
            print "cards", action.hinted_card_pos,
            print "are",
            print action.number if action.number is not None else action.color
        
        print
    
    
    def log_status(self):
        print "Hands:"
        for player in self.players:
            print "    Player %d" % player.id, player.hand
        print "Board:",
        for color in Card.COLORS:
            print colored("%d" % self.board[color], Card.PRINTABLE_COLORS[color]),
        print
        print "Hints: %d    Lives: %d    Deck: %d    Score: %d" % (self.hints, self.lives, len(self.deck), sum(self.board.itervalues()))
        if self.last_round:
            print "This is the last round (player %d plays last on turn %d)" % (self.last_player.id, self.last_turn)
        
        print

    
    def run_game(self):
        end_game = False
        current_player = self.players[0]
        
        self.log_status()
        
        while not end_game:
            raw_input()
            
            # do turn
            turn, end_game = self.run_turn(current_player)
            
            # log turn and status
            self.log_turn(turn, current_player)
            self.log_status()
            
            # inform all players
            for player in self.players:
                player.feed_turn(turn)
            
            # store turn
            self.turns.append(turn)
            
            # change current player
            current_player = current_player.next_player()
        
        




