

class Action:
    
    PLAY = 'Play'
    DISCARD = 'Discard'
    HINT = 'Hint'
    
    TYPES = [PLAY, DISCARD, HINT]
    
    
    def __init__(self, type, card_pos=None, player=None, color=None, number=None):
        assert type in self.TYPES
        
        if type in [self.PLAY, self.DISCARD]:
            assert card_pos is not None
            assert player is None
            assert color is None
            assert number is None
        
        elif type == self.HINT:
            assert card_pos is None
            assert player is not None
            assert color is not None and number is None or color is None and number is not None
        
        
        self.type = type
        self.card_pos = card_pos
        self.player = player
        self.color = color
        self.number = number
        
        if type == self.HINT:
            self.hinted_card_pos = [i for (i, card) in enumerate(player.hand) if card is not None and card.number == number or card.color == color]
            assert len(self.hinted_card_pos) > 0


    def __repr__(self):
        res = self.type
        if self.type in [self.PLAY, self.DISCARD]:
            res += " card %d" % self.card_pos
        return res

