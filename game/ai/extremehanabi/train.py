import sys, os, copy, random
from collections import namedtuple, Counter
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from game.ai.extremehanabi.action_manager import PolicyNetwork, ActionManager
from game.game import Game
from game.action import Action
from game.card import Card

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def random_index(self):
        return random.randint(0, len(self)-1)

    def push(self, *args):
        self.memory.append(Transition(*args))
        if len(self) > self.capacity:
            del self.memory[self.random_index()]

    def sample(self, batch_size=32):
        if len(self) < self.capacity:
            return None
        res = random.sample(self.memory, batch_size)
        
        states = torch.stack([transition.state for transition in res])
        actions = torch.stack([transition.action for transition in res])
        next_states = torch.stack([transition.next_state for transition in res])
        rewards = torch.stack([transition.reward for transition in res])
        
        return states, actions, next_states, rewards
    
    def get(self):
        if len(self) < self.capacity:
            return None
        
        i = self.random_index()
        x = self.memory[i]
        # del self.memory[i]
        return x
        

    def __len__(self):
        return len(self.memory)


"""
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
    for (log_prob, value, _, _), r in zip(saved_actions, rewards):
        reward = r - value.data[0]
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
    optimizer.zero_grad()
    loss = torch.cat(policy_losses).sum() + torch.cat(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]
"""

def optimize_model(model, optimizer, memory, batch_size=32):
    res = memory.sample(batch_size)
    if res is None:
        return
    
    states, actions, next_states, rewards = res
    
    action_scores, state_values = model(Variable(states))
    _, next_state_values = model(Variable(next_states))
    
    R = next_state_values + Variable(rewards) # obtained reward
    V = state_values  # expected reward
    
    # probs = action_scores.exp()
    # m = Categorical(probs) # probability distribution of actions
    # print action_scores, actions
    
    # compute log_prob of chosen actions
    log_probs = torch.masked_select(action_scores, mask=actions).unsqueeze(1)
    
    L_actor = - log_probs * (R-V).detach()
    L_critic = (R-V)**2
    
    optimizer.zero_grad()
    loss = (L_actor + L_critic).sum() / batch_size
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the network.')
    parser.add_argument('-c', '--clear', action='store_true', help='clear the network before training')
    args = parser.parse_args()
    
    if not args.clear and os.path.isfile("model.tar"):
        # load model from file
        model = torch.load("model.tar")
        print >> sys.stderr, "Loaded model from file"
    else:
        model = PolicyNetwork()
        print >> sys.stderr, "Created new model"
    
    optimizer = optim.Adam(model.parameters(), lr=4e-4)
    memory = ReplayMemory(200)
    
    running_avg = 0.0
    
    old_reward = None
    old_action = None
    old_state = None
    
    try:
        iteration = 0
        while True:
            game = Game(num_players=5, ai="extremehanabi", ai_params={'model': model, 'training': True}, strategy_log=(random.randint(0,10) == -1))
            game.setup()
            
            # train only on good decks
            if game.deck[0].color == Card.RAINBOW and game.deck[0].number < 5:
                continue
            
            chosen_actions = []
            previous_score = game.get_current_score()
            
            for player, turn in game.run_game():
                score = game.get_current_score()
                lives = game.lives
                
                reward = score - previous_score
                
                # penalize wrong action
                """
                if model.saved_action.chosen_action == ActionManager.HINT and turn.action.type != Action.HINT and game.hints == 0:
                    reward -= 1.0
                
                if model.saved_action.chosen_action == ActionManager.PLAY and turn.action.type != Action.PLAY and not game.last_round:
                    reward -= 1.0
                """
                
                reward = float(reward) / 30.0
                
                previous_score = score
                # model.rewards.append(reward)
                
                chosen_actions.append(model.saved_action.chosen_action)
                
                # store previous transition in memory
                if old_state is not None:
                    memory.push(old_state, old_action, model.saved_action.state, old_reward)
                
                old_reward = torch.Tensor([reward])
                old_state = model.saved_action.state
                old_action = model.saved_action.action_variable
                
                # Perform one step of the optimization (on the target network)
                optimize_model(model, optimizer, memory)
            
            # TODO add last turn
            
            print Counter(chosen_actions)
            print game.statistics
            running_avg = 0.95 * running_avg + 0.05 * game.statistics.score
            print "Running average score: %.2f" % running_avg
            
            print "Memory size:", len(memory)
            
            iteration += 1
            
            if iteration % 10 == 0:
                torch.save(model, "model.tar")
            
            # finish_episode(model, optimizer)

    except KeyboardInterrupt:
        print "Interrupted after %d iterations" % iteration
        torch.save(model, "model.tar")
