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
        raise Exception("Called init method of generic action")
    
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
    def __init__(self, player_id, color=None, number=None):
        self.type = self.HINT
        self.player_id = player_id
        
        assert color is not None and number is None or color is None and number is not None
        self.color = color
        self.number = number
        self.value = color if color is not None else number
        self.hint_type = self.COLOR if color is not None else self.NUMBER
    
    def __repr__(self):
        return "Play card %d" % self.card_pos
    
    def apply(self, game):
        # populate other fields, using information from the game
        super(HintAction, self).apply(game)
        
        player = game.players[self.player_id]
        self.cards_pos = [i for (i, card) in enumerate(player.hand) if card is not None and (card.number == self.number or card.color == self.color)]
        assert len(self.cards_pos) > 0


