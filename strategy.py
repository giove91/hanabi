import random
from collections import namedtuple

from action import Action
from card import Card, deck


Knowledge = namedtuple("Knowledge", "color number")


class IndirectHintsManager:
    def __init__(self, num_players, k):
        self.num_players = num_players
        self.k = k
        self.knowledge = [[Knowledge(color=False, number=False) for j in xrange(k)] for i in xrange(num_players)]


    def choose_card(self, player_id, target_id, turn, number_hint):
        # choose which of the target's cards receive a hint from the current player in the given turn
        possible_cards = [card_pos for (card_pos, kn) in enumerate(self.knowledge[target_id]) if not (kn.number if number_hint else kn.color)]
        
        if len(possible_cards) == 0:
            # do not give hints
            return None
        
        n = turn * 11**3 + (1 if number_hint else 0) * 11**2 + player_id * 11 + target_id
        
        return possible_cards[n % len(possible_cards)]



class Strategy:
    """
    An instance of this class represents a player's strategy.
    It only has the knowledge of that player, and it must make decisions.
    """
    
    def __init__(self):
        self.COLORS_TO_NUMBERS = {color: i for (i, color) in enumerate(Card.COLORS)}
    
    
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
        
        # next type of indirect hint to give
        self.number_hint = True
        
        # indirect hints manager
        self.indirect_hints_manager = IndirectHintsManager(num_players, k)
    
    
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
            # someone else gave a hint!
            
            if action.player_id == self.id:
                # they gave me a hint!
                
                if player_id == 4:
                    # hint on relevant cards
                    for i in action.hinted_card_pos:
                        self.relevant[i] = True
                    self.log("updated relevant cards: " + self.relevant.__repr__())
                
                # process direct hint
                for (i, p) in enumerate(self.possibilities):
                    for card in self.full_deck:
                        if not card.matches_hint(action, i) and card in p:
                            # self.log("removing card " + card.__repr__() + " from position %d due to hint" % i)
                            p.remove(card)
            
            if player_id < 4:
                # process indirect hint
                n = action.number if action.number_hint else self.COLORS_TO_NUMBERS[action.color]
                card_pos = player_id
                modulo = Card.NUM_NUMBERS if action.number_hint else Card.NUM_COLORS
                
                # n should be the sum of the visible cards in position card_pos,
                # so I can compute something about my card
                
                involved_cards = [hand[card_pos] for (i, hand) in self.hands.iteritems() if i != player_id]
                # self.log("detected indirect hint, cards: " + involved_cards.__repr__())
                m = sum(card.number if action.number_hint else self.COLORS_TO_NUMBERS[card.color] for card in involved_cards)
                my_value = (n - m) % modulo
                
                number = my_value if action.number_hint else None
                if number == 0:
                    number = 5
                color  = Card.COLORS[my_value] if not action.number_hint else None
                
                # self.log("thanks to indirect hint, understood that card %d has " % card_pos + ("number %d" % number if action.number_hint else "color %s" % color))
                
                p = self.possibilities[card_pos]
                for card in self.full_deck:
                    if not card.matches(color=color, number=number) and card in p:
                        # print card
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
        
        
        for (card_pos, p) in enumerate(self.possibilities):
            if len(p) < 5:
                self.log("in position %d the possibilities are" % card_pos + p.__repr__())
        
        # check for playable cards in my hand
        for (card_pos, p) in enumerate(self.possibilities):
            if all(card.playable(self.board) for card in p) and len(p) > 0:
                # the card in this position is surely playable!
                # play the card
                return Action(Action.PLAY, card_pos=card_pos)
        
        
        if self.hints == 0:
            # discard card
            # TODO: choose wisely the card to be discarded
            return Action(Action.DISCARD, card_pos=random.randint(0, 3))
        
        
        # try to give hint
        if self.id < 4:
            # try to give indirect hint
            
            # compute sum of visible cards in position card_pos
            modulo = Card.NUM_NUMBERS if self.number_hint else Card.NUM_COLORS
            card_pos = self.id
            involved_cards = [hand[card_pos] for (i, hand) in self.hands.iteritems()]
            m = sum(card.number if self.number_hint else self.COLORS_TO_NUMBERS[card.color] for card in involved_cards) % modulo
            
            number = m if self.number_hint else None
            if number == 0:
                number = 5
            color  = Card.COLORS[m] if not self.number_hint else None
            
            # search for a card matching the hint
            player_id = None
            for (i, hand) in self.hands.iteritems():
                for card in hand:
                    if card is not None and card.matches(color=color, number=number):
                        player_id = i
            
            if player_id is None:
                # failed to find a suitable card
                # discard card
                self.log("failed to give an indirect hint about value " + number.__repr__() + " " + color.__repr__())
                # TODO: choose wisely the card to be discarded
                return Action(Action.DISCARD, card_pos=random.randint(0, 3))
            
            else:
                # found a suitable card
                
                # change type of next indirect hint
                self.number_hint = False
                
                # finally give indirect hint
                # self.log("giving indirect hint")
                return Action(Action.HINT, player_id=player_id, color=color, number=number)
        
        else:
            # try to give hint on some relevant card
            # TODO
            return Action(Action.DISCARD, card_pos=random.randint(0, 3))


    
    def log(self, message):
        print "Player %d: %s" % (self.id, message)




