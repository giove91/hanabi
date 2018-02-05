#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools
import copy
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


from ...action import Action, PlayAction, DiscardAction, HintAction
from ...card import Card, CardAppearance


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.affine1 = nn.Linear(655, 128)
        self.affine2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, 3)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=0), state_values


SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'chosen_action'])


class ActionManager(object):
    
    # possible actions to choose
    PLAY = 'Play'
    DISCARD = 'Discard'
    HINT = 'Hint'
    ACTIONS = [PLAY, DISCARD, HINT]
    
    def __init__(self, strategy):
        self.strategy = strategy    # my strategy object
        
        # copy something from the strategy
        self.id = strategy.id
        self.num_players = strategy.num_players
        self.k = strategy.k
        self.possibilities = strategy.possibilities
        self.full_deck = strategy.full_deck
        self.board = strategy.board
        self.knowledge = strategy.knowledge
        self.hands = self.strategy.hands
        
        self.COLORS_TO_NUMBERS = {color: i for (i, color) in enumerate(Card.COLORS)}
        
        if 'model' in strategy.params:
            # use the neural network given as a parameter
            self.model = strategy.params['model']
        else:
            self.model = PolicyNetwork()
    
    
    def log(self, message):
        self.strategy.log(message)
    
    
    def get_state(self):
        state = []
        
        # deck size
        state.append(self.strategy.deck_size * 2.0 / 55.0 - 1.0)
        
        # hints
        state.append(self.strategy.hints * 2.0 / 8.0 - 1.0)
        
        # lives
        state.append(self.strategy.lives * 2.0 / 3.0 - 1.0)
        
        # board
        for color in Card.COLORS:
            for i in xrange(Card.NUM_NUMBERS+1):
                state.append(1 if self.strategy.board[color] == i else 0)
        
        # my hand
        for card_pos in xrange(self.k):
            for color in Card.COLORS:
                for number in xrange(1, Card.NUM_NUMBERS+1):
                    state.append(float(self.possibilities[card_pos][CardAppearance(color, number)]) / sum(self.possibilities[card_pos].itervalues()))
        
        # other hands
        for i in range(self.id+1, self.num_players) + range(self.id):
            for card_pos in xrange(self.k):
                for color in Card.COLORS:
                    for number in xrange(1, Card.NUM_NUMBERS+1):
                        state.append(1 if self.hands[i][card_pos].matches(color, number) else 0)
                
                # knowledge
                state.append(1 if self.knowledge[i][card_pos].knows_exactly() or self.knowledge[i][card_pos].useless else 0)
        
        return torch.Tensor(state)
    
    
    def select_action(self):
        state = self.get_state()
        probs, state_value = self.model(Variable(state))
        m = Categorical(probs)
        action = m.sample()
        chosen_action = self.ACTIONS[action.data[0]]
        self.model.saved_actions.append(SavedAction(m.log_prob(action), state_value, chosen_action))
        # return action.data[0]
        return chosen_action



