import sys, os
from collections import namedtuple, Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from game.ai.extremehanabi.action_manager import PolicyNetwork
from game.game import Game

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



def finish_episode(model, optimizer):
    R = 0   # reward
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + R   # gamma = 1
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for (log_prob, value, _), r in zip(saved_actions, rewards):
        reward = r - value.data[0]
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
    optimizer.zero_grad()
    loss = torch.cat(policy_losses).sum() + torch.cat(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


if __name__ == '__main__':
    
    model = PolicyNetwork()
    optimizer = optim.Adam(model.parameters(), lr=3e-2)
    
    for i_episode in xrange(1000):
        game = Game(num_players=5, ai="extremehanabi", ai_params={'model': model})
        game.setup()
        
        previous_score = game.get_current_score()
        
        for player, turn in game.run_game():
            score = game.get_current_score()
            lives = game.lives
            
            reward = score - previous_score
            if lives == 0:
                # the game is lost
                reward -= 50
            
            # TODO penalizzare azioni errate?
            
            previous_score = score
            model.rewards.append(reward)
        
        print Counter(saved_action.chosen_action for saved_action in model.saved_actions)
        print game.statistics
        finish_episode(model, optimizer)

