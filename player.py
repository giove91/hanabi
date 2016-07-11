from card import Card
from action import Action


class Player:
    def __init__(self, id, game, hand):
        # my id (order of play)
        self.id = id
        
        # game
        self.game = game
        
        # initial hand of cards
        self.hand = hand
    
    
    def __eq__(self, other):
        return self.id == other.id
    
    
    def next_player(self):
        return self.game.players[(self.id + 1) % self.game.num_players]
    
    
    def get_turn_action(self):
        # choose action for this turn
        # TODO
        card_pos = min(i for (i, card) in enumerate(self.hand) if card is not None)
        # return Action(Action.PLAY, card_pos=card_pos)
        return Action(Action.HINT, player=self.next_player, number=1)
    
    
    def feed_turn(self):
        # get informed about what happened during a turn
        # TODO
        pass




