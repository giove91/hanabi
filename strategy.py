import random

from action import Action
from card import Card, deck


class Knowledge:
    def __init__(self, color=False, number=False):
        self.color = color
        self.number = number
    
    def __repr__(self):
        return ("C" if self.color else "-") + ("N" if self.number else "-")


class IndirectHintsManager:
    def __init__(self, num_players, k, id, strategy):
        self.num_players = num_players
        self.k = k
        self.id = id    # my player id
        self.strategy = strategy    # my strategy object
        self.knowledge = [[Knowledge(color=False, number=False) for j in xrange(k)] for i in xrange(num_players)]
        
        self.COLORS_TO_NUMBERS = {color: i for (i, color) in enumerate(Card.COLORS)}


    def reset_knowledge(self, player_id, card_pos, new_card_exists):
        self.knowledge[player_id][card_pos] = Knowledge(False, False) if new_card_exists else Knowledge(True, True)


    def choose_card(self, player_id, target_id, turn, number_hint):
        # choose which of the target's cards receive a hint from the current player in the given turn
        possible_cards = [card_pos for (card_pos, kn) in enumerate(self.knowledge[target_id]) if not (kn.number if number_hint else kn.color)]
        
        if len(possible_cards) == 0:
            # do not give hints
            return None
        
        n = turn * 11**3 + (1 if number_hint else 0) * 119 + player_id * 11 + target_id
        
        return possible_cards[n % len(possible_cards)]
    
    
    def choose_all_cards(self, player_id, turn, number_hint):
        # choose all cards that receive hints from the given player
        return {target_id: self.choose_card(player_id, target_id, turn, number_hint) for target_id in xrange(self.num_players) if target_id != player_id and self.choose_card(player_id, target_id, turn, number_hint) is not None}
    
    
    def receive_hint(self, player_id, action):
        number_hint = action.number_hint
        cards_pos = self.choose_all_cards(player_id, action.turn, number_hint)
        
        # update knowledge
        for (target_id, card_pos) in cards_pos.iteritems():
            kn = self.knowledge[target_id][card_pos]
            if number_hint:
                kn.number = True
            else:
                kn.color = True
        
        for card_pos in action.hinted_card_pos:
            kn = self.knowledge[action.player_id][card_pos]
            if number_hint:
                kn.number = True
            else:
                kn.color = True
        
        # decode my hint
        if self.id in cards_pos:
            n = action.number if number_hint else self.COLORS_TO_NUMBERS[action.color]
            my_card_pos = cards_pos[self.id]
            modulo = Card.NUM_NUMBERS if number_hint else Card.NUM_COLORS
            
            involved_cards = [hand[cards_pos[i]] for (i, hand) in self.strategy.hands.iteritems() if i != player_id and i in cards_pos]
            
            m = sum(card.number if number_hint else self.COLORS_TO_NUMBERS[card.color] for card in involved_cards)
            my_value = (n - m) % modulo
            
            number = my_value if number_hint else None
            if number == 0:
                number = 5
            color = Card.COLORS[my_value] if not number_hint else None
            
            return my_card_pos, color, number
        
        else:
            # no hint (apparently I already know everything)
            return None
    
    
    def compute_hint(self, turn, number_hint):
        # returns the color or the number we need to give a hint about
        cards_pos = self.choose_all_cards(self.id, turn, number_hint)
        
        if len(cards_pos) == 0:
            # the other players already know everything
            return None
        
        # compute sum of visible cards in the given positions
        modulo = Card.NUM_NUMBERS if number_hint else Card.NUM_COLORS
        involved_cards = [hand[cards_pos[i]] for (i, hand) in self.strategy.hands.iteritems() if i in cards_pos]
        m = sum(card.number if number_hint else self.COLORS_TO_NUMBERS[card.color] for card in involved_cards) % modulo
        
        number = m if number_hint else None
        if number == 0:
            number = 5
        color = Card.COLORS[m] if not number_hint else None
        
        return color, number
        
        



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
        
        # remove cards of other players from possibilities
        self.update_possibilities()
        
        # indirect hints manager
        self.indirect_hints_manager = IndirectHintsManager(num_players, k, id, self)
    
    
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
        if action.type in [Action.PLAY, Action.DISCARD]:
            # reset knowledge of the player
            new_card = self.my_hand[action.card_pos] if player_id == self.id else self.hands[player_id][action.card_pos]
            self.indirect_hints_manager.reset_knowledge(player_id, action.card_pos, new_card is not None)
            
            if player_id == self.id:
                # check for my new card
                self.possibilities[action.card_pos] = set(self.full_deck) if self.my_hand[action.card_pos] is not None else set()
        
        elif action.type == Action.HINT:
            # someone gave a hint!
            
            if action.player_id == self.id:
                # they gave me a hint!
                
                # process direct hint
                for (i, p) in enumerate(self.possibilities):
                    for card in self.full_deck:
                        if not card.matches_hint(action, i) and card in p:
                            # self.log("removing card " + card.__repr__() + " from position %d due to hint" % i)
                            p.remove(card)
            
            
            # process indirect hint
            res = self.indirect_hints_manager.receive_hint(player_id, action)
            
            if res is not None:
                card_pos, color, number = res
                self.log("thanks to indirect hint, understood that card %d has " % card_pos + ("number %d" % number if action.number_hint else "color %s" % color))
            
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
        
        
        print "Knowledge"
        for i in xrange(self.num_players):
            print "Player %d:" % i,
            for card_pos in xrange(self.k):
                print self.indirect_hints_manager.knowledge[i][card_pos],
            print
        print
        
        
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
        
        
        # try to give indirect hint
        # TODO: choose number_hint
        number_hint = random.randint(0, 1) == 0
        res = self.indirect_hints_manager.compute_hint(self.turn, number_hint)
        if res is not None:
            color, number = res
        
            # search for a card matching the hint
            # TODO: do it better
            player_id = None
            for (i, hand) in self.hands.iteritems():
                for card in hand:
                    if card is not None and card.matches(color=color, number=number):
                        player_id = i
            
            if player_id is not None:
                # found a suitable card
                # finally give indirect hint
                # self.log("giving indirect hint")
                return Action(Action.HINT, player_id=player_id, color=color, number=number)
        
        # failed to give indirect hint
        # discard card
        self.log("failed to give an indirect hint about value " + number.__repr__() + " " + color.__repr__())
        # TODO: choose wisely the card to be discarded
        return Action(Action.DISCARD, card_pos=random.randint(0, 3))



    
    def log(self, message):
        print "Player %d: %s" % (self.id, message)




