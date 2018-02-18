#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import itertools
import copy
from collections import namedtuple
import random

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
        INPUT_SIZE = 6
        SIZE = 10
        self.affine1 = nn.Linear(INPUT_SIZE, SIZE)
        self.batch_norm1 = nn.BatchNorm1d(SIZE)       
        self.dropout1 = nn.Dropout()
         
        self.affine2 = nn.Linear(SIZE, SIZE)
        self.batch_norm2 = nn.BatchNorm1d(SIZE)
        self.dropout2 = nn.Dropout()
        
        self.action_head = nn.Linear(SIZE, 3)
        self.value_head = nn.Linear(SIZE, 1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        self.saved_action = None

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.affine1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.batch_norm2(self.affine2(x)))
        x = self.dropout2(x)
        
        action_score = self.action_head(x)
        state_value = self.value_head(x)
        
        return self.logsoftmax(action_score), state_value


SavedAction = namedtuple('SavedAction', ['value', 'chosen_action', 'state', 'action_variable'])


class ActionManager(object):
    
    # possible actions to choose
    PLAY = 'Play'
    DISCARD = 'Discard'
    HINT = 'Hint'
    ACTIONS = [PLAY, DISCARD, HINT]
    
    MODEL_FILENAME = 'game/ai/extremehanabi/model.tar'
    
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
        self.discard_pile = self.strategy.discard_pile
        
        self.COLORS_TO_NUMBERS = {color: i for (i, color) in enumerate(Card.COLORS)}
        
        if 'model' in strategy.params:
            # use the neural network given as a parameter
            self.model = strategy.params['model']
        elif os.path.isfile(self.MODEL_FILENAME):
            self.model = torch.load(self.MODEL_FILENAME)
        else:
            raise Exception("Model not found")
            self.model = PolicyNetwork()
    
    
    def log(self, message):
        self.strategy.log(message)
    
    
    def get_state(self):
        state = []
        
        # deck size
        state.append(self.strategy.deck_size * 2.0 / 55.0 - 1.0)
        
        # hints
        # for i in xrange(9):
        #     state.append(1.0 if self.strategy.hints == i else 0.0)
        # state.append(self.strategy.hints * 2.0 / 8.0 - 1.0)
        state.append(1.0 if self.strategy.hints > 0 else 0.0)
        
        # score
        state.append(sum(self.strategy.board.itervalues()) * 2.0 / 30.0 - 1.0)
        
        # do I have a 100% playable card?
        state.append(any(len(p) > 0 and all(card.playable(self.board) for card in p) for (card_pos, p) in enumerate(self.possibilities)))
        
        # do I have a 100% useless card?
        state.append(any(len(p) > 0 and all(not card.useful(self.board, self.full_deck, self.discard_pile) for card in p) for (card_pos, p) in enumerate(self.possibilities)))
        
        # do I have a 100% non-relevant card?
        state.append(any(len(p) > 0 and all(not card.relevant(self.board, self.full_deck, self.discard_pile) for card in p) for (card_pos, p) in enumerate(self.possibilities)))
        
        """
        # lives
        # state.append(self.strategy.lives * 2.0 / 3.0 - 1.0)
        
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
        """
        
        return torch.Tensor(state)
    
    
    def select_action(self):
        state = self.get_state()
        self.model.eval() # evaluation mode
        
        action_score, state_value = self.model(Variable(state.unsqueeze(0)))
        probs = action_score.exp()
        self.log("Probabilities: %r" % list(probs.data))
        self.log("Value: %r" % float(state_value.data))
        """
        if 'training' in self.strategy.params and self.strategy.params['training']:
            # sample from probability distribution
            m = Categorical(probs)
            action = m.sample()
            chosen_action = self.ACTIONS[action.data[0]]
            action_mask = Variable(torch.ByteTensor([1 if i==action.data[0] else 0 for i in xrange(len(self.ACTIONS))]))
            self.model.saved_action = SavedAction(state_value, chosen_action, state, action_mask)
            return chosen_action
        """
        
        if 'training' in self.strategy.params and random.random() < self.strategy.params['training']:
            # choose at random
            action = random.randint(0, len(self.ACTIONS)-1)
        
        else:
            # choose best action
            probs = probs.data.numpy()
            action = probs.argmax()
        
        chosen_action = self.ACTIONS[action]
        action_mask = Variable(torch.ByteTensor([1 if i==action else 0 for i in xrange(len(self.ACTIONS))]))
        self.model.saved_action = SavedAction(state_value, chosen_action, state, action_mask)
        return chosen_action



