#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import itertools
import copy
from collections import namedtuple, Counter
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
        
        self.saved_action = None
        
        self.conv1 = nn.Conv2d(14, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(16)
        
        self.affine = nn.Linear(16*6, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout()
        
        self.action_head = nn.Linear(16, 9)
        self.value_head = nn.Linear(16, 1)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        B = x.size()[0]
        
        # Bx6x5x5xN -> (B*6)x5x5xN -> (B*6)xNx5x5
        x = x.view((B*6, 5, 5, -1)).transpose(1, 3)
        
        # convolutional layers (conv -> batch normalization -> relu)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # (B*6)x1x1xN' -> Bx(6*N')
        x = x.view((B, -1))
        
        x = F.relu(self.bn4(self.affine(x)))
        x = self.dropout(x)
        
        action_score = self.action_head(x)
        state_value = self.value_head(x)
        
        return self.logsoftmax(action_score), state_value


SavedAction = namedtuple('SavedAction', ['value', 'chosen_action', 'state', 'action_variable'])


class ActionManager(object):
    
    # possible actions to choose
    PLAY = 'Play'
    DISCARD = 'Discard'
    HINT = 'Hint'
    NUM_ACTIONS = 9 # 0,...,3: play; 4,...7: discard; 8: hint
    
    MODEL_FILENAME = 'game/ai/extremehanabi/model.tar'
    
    def __init__(self, strategy):
        self.strategy = strategy    # my strategy object
        
        # copy something from the strategy
        self.id = strategy.id
        self.num_players = strategy.num_players
        self.k = strategy.k
        self.possibilities = strategy.possibilities
        self.full_deck = strategy.full_deck
        self.knowledge = strategy.knowledge
        
        self.COLORS_TO_NUMBERS = {color: i for (i, color) in enumerate(Card.COLORS)}
        
        self.state_shape = (len(Card.COLORS), Card.NUM_NUMBERS, self.num_players, -1)
        
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
        
        discard_pile = Counter(self.strategy.discard_pile)
        
        for color in Card.COLORS:
            for number in xrange(1, Card.NUM_NUMBERS+1):
                for player in range(self.id, self.num_players) + range(self.id):
                    if player == self.id:
                        # my possibilities
                        for (card_pos, p) in enumerate(self.possibilities):
                            state.append(1.0 if any(card.matches(color=color, number=number) for card in p) else 0.0)
                        
                        # (fake) my cards -- needed to fill the tensor's shape
                        for card_pos in xrange(self.k):
                            state.append(0.0)
                    
                    else:
                        # others' possibilities
                        for card_pos in xrange(self.k):
                            state.append(1.0 if self.knowledge[player][card_pos].compatible(
                                self.strategy.hands[player][card_pos],
                                color,
                                number,
                                self.strategy.board,
                                self.full_deck,
                                self.strategy.discard_pile
                            ) else 0.0)
                        
                        # others' cards
                        for (card_pos, card) in enumerate(self.strategy.hands[player]):
                            state.append(1.0 if card is not None and card.matches(color=color, number=number) else 0.0)
                    
                    # board
                    state.append(1.0 if self.strategy.board[color] >= number else 0.0)
                    
                    # discard pile
                    state.append(float(discard_pile[CardAppearance(color=color, number=number)]))
                    
                    # score
                    # state.append(sum(self.strategy.board.itervalues()) * 2.0 / 30.0 - 1.0)
                    
                    # hints
                    state.append(self.strategy.hints * 2.0 / 8.0 - 1.0)
                    
                    # deck size
                    state.append(self.strategy.deck_size * 2.0 / 35.0 - 1.0)
                    
                    # num_players + 1 - remaining turns (if deck size == 0)
                    state.append(float(self.num_players + 1 - self.strategy.last_turn + self.strategy.turn) \
                        if self.strategy.last_turn is not None else 0.0)
                    
                    # lives
                    state.append(self.strategy.lives * 2.0 / 3.0 - 1.0)
        
        """
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
        state.append(any(len(p) > 0 and all(not card.useful(self.board, self.full_deck, self.strategy.discard_pile) for card in p) for (card_pos, p) in enumerate(self.possibilities)))
        
        # do I have a 100% non-relevant card?
        state.append(any(len(p) > 0 and all(not card.relevant(self.board, self.full_deck, self.strategy.discard_pile) for card in p) for (card_pos, p) in enumerate(self.possibilities)))
        """
        
        res = torch.Tensor(state)
        res = res.view(self.state_shape)
        
        return res
    
    
    def select_action(self):
        state = self.get_state()
        self.model.eval() # evaluation mode
        
        action_score, state_value = self.model(Variable(state.unsqueeze(0)))
        probs = action_score.exp()
        
        import random
        if random.random() < 0.01:
            print
            print probs
        
        self.log("Probabilities: %r" % list(probs.data))
        self.log("Value: %r" % float(state_value.data))
        
        if 'training' in self.strategy.params:
            # sample from probability distribution
            m = Categorical(probs)
            action = m.sample().data[0]
        
        else:
            # choose best action
            probs = probs.data.numpy()
            action = probs.argmax()
        
        # interpret chosen action
        if 0 <= action < 4:
            chosen_action = (self.PLAY, action)
        elif 4 <= action < 8:
            chosen_action = (self.DISCARD, action-4)
        else:
            chosen_action = (self.HINT, None)
        
        action_mask = Variable(torch.ByteTensor([1 if i==action else 0 for i in xrange(self.NUM_ACTIONS)]))
        self.model.saved_action = SavedAction(state_value, chosen_action, state, action_mask)
        return chosen_action
        
        """
        if 'training' in self.strategy.params and random.random() < self.strategy.params['training']:
            # choose at random
            action = random.randint(0, len(self.ACTIONS)-1)
        """



