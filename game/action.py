#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Action(object):
    """
    Generic action.
    """
    PLAY = 'Play'
    DISCARD = 'Discard'
    HINT = 'Hint'
    
    COLOR = 'C'
    NUMBER = 'N'
    
    TYPES = [PLAY, DISCARD, HINT]
    HINT_TYPES = [COLOR, NUMBER]
    
    
    def __init__(self, type, card_pos=None, player_id=None, color=None, number=None):
        raise NotImplementedError
    
    def apply(self, game):
        # populate other fields, using information from the game
        self.turn = game.get_current_turn()



class PlayAction(Action):
    """
    Action of type PLAY.
    """
    def __init__(self, card_pos):
        self.type = self.PLAY
        self.card_pos = card_pos
    
    def __repr__(self):
        return "Play card %d" % self.card_pos


class DiscardAction(Action):
    """
    Action of type DISCARD.
    """
    def __init__(self, card_pos):
        self.type = self.DISCARD
        self.card_pos = card_pos
    
    def __repr__(self):
        return "Discard card %d" % self.card_pos


class HintAction(Action):
    """
    Action of type HINT.
    """
    def __init__(self, player_id, color=None, number=None, hint_type=None, value=None):
        """
        A HintAction can be constructed giving the color or the number, or giving the hint type and the value.
        """
        self.type = self.HINT
        self.player_id = player_id
        
        if color is not None or number is not None:
            assert color is not None and number is None or color is None and number is not None
            assert hint_type is None and value is None
            self.color = color
            self.number = number
            self.value = color if color is not None else number
            self.hint_type = self.COLOR if color is not None else self.NUMBER
        else:
            assert hint_type is not None and value is not None
            assert hint_type in self.HINT_TYPES
            self.hint_type = hint_type
            self.value = value
            if hint_type == Action.COLOR:
                self.color = value
                self.number = None
            else:
                self.color = None
                self.number = value
    
    
    def __repr__(self):
        return "Hint to player %d about %r" % (self.player_id, self.value)
    
    def apply(self, game):
        # populate other fields, using information from the game
        super(HintAction, self).apply(game)
        
        player = game.players[self.player_id]
        self.cards_pos = [i for (i, card) in enumerate(player.hand) if card is not None and (card.number == self.number or card.color == self.color)]
        assert len(self.cards_pos) > 0


