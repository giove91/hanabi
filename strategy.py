from action import Action
from card import Card, deck


class Strategy:
    """
    An instance of this class represents a player's strategy.
    It only has the knowledge of that player, and it must make decisions.
    """
    
    def __init__(self):
        pass
    
    
    def initialize(self, id, num_players, k, hands, board, discard_pile):
        # to be called once before the beginning
        self.id = id
        self.num_players = num_players
        self.k = k  # number of cards per hand
        self.hands = hands  # hands of other players
        self.board = board
        self.discard_pile = discard_pile
        
        # store a copy of the full deck
        self.full_deck = deck()
        
        # for each of my card, I store its possibilities
        self.possibilities = [set(self.full_deck) for i in xrange(self.k)]
        self.relevant = [False] * self.k
        
        # remove cards of other players from possibilities
        self.update_possibilities()
    
    
    def update(self, hints, lives, my_hand, turn, last_turn):
        # to be called every turn
        self.hints = hints
        self.lives = lives
        self.my_hand = my_hand
        self.turn = turn
        self.last_turn = last_turn
    
    
    def visible_cards(self):
        for card in self.discard_pile:
            yield card
        
        for hand in self.hands.itervalues():
            for card in hand:
                yield card
    
    def update_possibilities(self):
        # update possibilities removing visible cards
        for card in self.visible_cards():
            for p in self.possibilities:
                if card in p:
                    p.remove(card)
    
    
    def next_player_id(self):
        return (self.id + 1) % self.num_players
    
    def other_players_id(self):
        return [i for i in xrange(self.num_players) if i != self.id]
    
    
    def feed_turn(self, player_id, action):
        if player_id == self.id and action.type in [Action.PLAY, Action.DISCARD]:
            # check for my new card
            self.possibilities[action.card_pos] = set(self.full_deck) if self.my_hand[action.card_pos] is not None else set()
            self.relevant[action.card_pos] = False
        
        elif player_id != self.id and action.type == Action.HINT:
            # someone gave a hint!
            
            if action.player_id == self.id:
                # they gave me a hint!
                
                if player_id == 4:
                    # hint on relevant cards
                    for i in action.hinted_card_pos:
                        self.relevant[i] = True
                    self.log("updated relevant cards: " + self.relevant.__repr__())
                
                for (i, p) in enumerate(self.possibilities):
                    for card in self.full_deck:
                        if not card.matches_hint(action, i) and card in p:
                            self.log("removing card " + card.__repr__() + " from position %d due to hint" % i)
                            p.remove(card)
        
        # update possibilities with visible cards
        self.update_possibilities()
    
    
    def get_turn_action(self):
        # strategy for a 5-player game
        assert self.num_players == 5
        assert self.k == 4
        
        # print list(self.visible_cards())
        # print self.possibilities
        
        # idea:
        # player i gives information about the cards in position i
        # player 4 gives information about relevant cards (e.g. unique cards, cards to be played)
        
        if self.hints > 0:
            return Action(Action.HINT, player_id=self.next_player_id(), number=1)
        else:
            card_pos = 0
            return Action(Action.DISCARD, card_pos=card_pos)

    
    def log(self, message):
        print "Player %d: %s" % (self.id, message)




