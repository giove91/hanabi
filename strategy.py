from action import Action
from card import Card


class Strategy:
    """
    An instance of this class represents a player's strategy.
    It only has the knowledge of that player, and it must make decisions.
    """
    
    def __init__(self):
        pass
    
    
    def initialize(self, id, num_players, hands, board, discard_pile):
        # to be called once before the beginning
        self.id = id
        self.num_players = num_players
        self.hands = hands  # hands of other players
        self.board = board
        self.discard_pile = discard_pile
    
    
    def update(self, hints, lives, my_hand, turn):
        # to be called every turn
        self.hints = hints
        self.lives = lives
        self.my_hand = my_hand
        self.turn = turn
    
    
    def next_player_id(self):
        return (self.id + 1) % self.num_players
    
    def other_players_id(self):
        return [i for i in xrange(self.num_players) if i != self.id]
    
    
    def feed_turn(self, player_id, action):
        print "%d receives %d" % (self.id, player_id), action
        # TODO
    
    
    def get_turn_action(self):
        if self.hints > 0:
            return Action(Action.HINT, player_id=self.next_player_id(), number=1)
        else:
            card_pos = 0
            return Action(Action.DISCARD, card_pos=card_pos)



