import random

from card import Card, deck
from player import Player



class Game:
    CARDS_PER_PLAYER = {2: 5, 3: 5, 4: 4, 5: 4}
    
    
    def __init__(self, num_players):
        self.num_players = num_players
        
        # compute number of cards per player
        self.k = self.CARDS_PER_PLAYER[num_players]
    
    
    def get_card_from_deck(self):
        assert len(self.deck) > 0
        return self.deck.pop()
    
    
    def setup(self):
        # construct deck
        self.deck = deck()
        
        # shuffle deck
        random.shuffle(self.deck)
        
        # initialize players, with initial hand of cards
        self.players = [Player([self.get_card_from_deck() for i in xrange(self.k)]) for i in xrange(self.num_players)]
        
        


