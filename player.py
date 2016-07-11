from action import Action


class Player:
    def __init__(self, id, hand):
        # my id (order of play)
        self.id = id
        
        # initial hand of cards
        self.hand = hand
    
    
    def __eq__(self, other):
        return self.id == other.id
    
    
    def get_turn_action(self):
        # choose action for this turn
        # TODO
        card_pos = min(i for (i, card) in enumerate(self.hand) if card is not None)
        return Action(Action.DISCARD, card_pos=card_pos)
    
    
    def feed_turn(self):
        # get informed about what happened during a turn
        # TODO
        pass


