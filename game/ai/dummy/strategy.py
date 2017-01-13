#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random

from ...action import Action, PlayAction, DiscardAction, HintAction
from ...card import Card, deck
from ...base_strategy import BaseStrategy


class Strategy(BaseStrategy):
    """
    An instance of this class represents a player's strategy.
    It only has the knowledge of that player, and it must make decisions.
    """
    
    def feed_turn(self, player_id, action):
        pass
    
    
    def get_turn_action(self):
        
        if self.hints > 0 and random.randint(0,2) == 0:
            # give random hint to the next player
            player_id = (self.id + 1) % self.num_players
            card = random.choice([card for card in self.hands[player_id] if card is not None])
            
            if random.randint(0,1) == 0:
                color = card.color
                number = None
            else:
                color = None
                number = card.number
            
            self.log("give some random hint")
            return HintAction(player_id, color=color, number=number)
        
        elif random.randint(0,1) == 0:
            # play random card
            card_pos = random.choice([c_pos for (c_pos, value) in enumerate(self.my_hand) if value is not None])
            self.log("play some random card")
            return PlayAction(card_pos)
        
        else:
            # discard random card
            card_pos = random.choice([c_pos for (c_pos, value) in enumerate(self.my_hand) if value is not None])
            self.log("discard some random card")
            return DiscardAction(card_pos)



