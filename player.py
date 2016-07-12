from card import Card
from action import Action
from strategy import Strategy


class Player:
    def __init__(self, id, game, hand):
        # my id (order of play)
        self.id = id
        
        # game
        self.game = game
        
        # initial hand of cards
        self.hand = hand
        
        # strategy object
        self.strategy = Strategy()
    
    
    def __eq__(self, other):
        return self.id == other.id
    
    
    def next_player(self):
        return self.game.players[(self.id + 1) % self.game.num_players]
    
    def other_players(self):
        return {i: player for (i, player) in enumerate(self.game.players) if player != self}
    
    
    def initialize(self):
        # called once after all players are created, before the game starts
        self.initialize_strategy()
    
    
    def initialize_strategy(self):
        self.strategy.initialize(
                id = self.id,
                num_players = self.game.num_players,
                k = self.game.k,
                hands = {i: player.hand for (i, player) in self.other_players().iteritems()},
                board = self.game.board,
                discard_pile = self.game.discard_pile
            )
    
    def update_strategy(self):
        self.strategy.update(
                hints = self.game.hints,
                lives = self.game.lives,
                my_hand = [0 if card is not None else None for (i, card) in enumerate(self.hand)],
                turn = self.game.get_current_turn(),
                last_turn = self.game.last_turn
            )
    
    
    def get_turn_action(self):
        # update strategy (in case this is the first turn)
        self.update_strategy()
        
        # choose action for this turn
        action = self.strategy.get_turn_action()
        action.apply(self.game)
        return action
    
    
    def feed_turn(self, turn):
        # update strategy
        self.update_strategy()
        
        # get informed about what happened during a turn
        self.strategy.feed_turn(turn.player.id, turn.action)


