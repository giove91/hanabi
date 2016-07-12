from action import Action


class Strategy:
    def __init__(self, id, num_players):
        self.id = id
        self.num_players = num_players
        
        self.hints = None
        self.lives = None
        self.turn = None
    
    
    def update(self, hints, lives, hands, my_hand, turn):
        self.hints = hints
        self.lives = lives
        self.hands = hands
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
