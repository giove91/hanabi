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
from ...card import Card


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.affine1 = nn.Linear(30, 128)
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
        return F.softmax(action_scores, dim=1), state_values


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class ActionManager(object):
    
    # possible actions to choose
    PLAY = 'P'
    DISCARD = 'D'
    HINT = 'H'
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
        
        self.COLORS_TO_NUMBERS = {color: i for (i, color) in enumerate(Card.COLORS)}
        
        self.model = PolicyNetwork()
    
    def log(self, message):
        self.strategy.log(message)

    def select_action():
        state = torch.rand(30)  # TODO: real state definition!
        probs, state_value = self.model(Variable(state))
        m = Categorical(probs)
        action = m.sample()
        model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        # return action.data[0]
        return ACTIONS[action]



