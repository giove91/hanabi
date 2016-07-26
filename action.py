#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Action:
    
    PLAY = 'Play'
    DISCARD = 'Discard'
    HINT = 'Hint'
    
    COLOR = 'C'
    NUMBER = 'N'
    
    TYPES = [PLAY, DISCARD, HINT]
    HINT_TYPES = [COLOR, NUMBER]
    
    
    def __init__(self, type, card_pos=None, player_id=None, color=None, number=None):
        assert type in self.TYPES
        
        if type in [self.PLAY, self.DISCARD]:
            assert card_pos is not None
            assert player_id is None
            assert color is None
            assert number is None
        
        elif type == self.HINT:
            assert card_pos is None
            assert player_id is not None
            assert color is not None and number is None or color is None and number is not None
        
        
        self.type = type
        self.card_pos = card_pos
        self.player_id = player_id
        self.color = color
        self.number = number
        
        if type == self.HINT:
            self.hint_type = self.COLOR if color is not None else self.NUMBER
    
    
    def apply(self, game):
        # populate other fields, using information from the game
        self.turn = game.get_current_turn()
        
        if self.type == self.HINT:
            player = game.players[self.player_id]
            self.hinted_card_pos = [i for (i, card) in enumerate(player.hand) if card is not None and (card.number == self.number or card.color == self.color)]
            assert len(self.hinted_card_pos) > 0
            
            # FIXME deprecated! use self.hint_type instead!
            self.number_hint = self.number is not None  # boolean variable to know if the hint is about numbers or colors


    def __repr__(self):
        res = self.type
        if self.type in [self.PLAY, self.DISCARD]:
            res += " card %d" % self.card_pos
        return res

