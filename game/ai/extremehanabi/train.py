import sys, os, copy, random, math
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
        # print self.memory[-1]
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
    model.train() # training mode
    res = memory.sample(batch_size)
    if res is None:
        return
    
    states, actions, next_states, rewards = res
    """
    print states,
    print actions,
    print next_states,
    print rewards
    """
    action_scores, state_values = model(Variable(states))
    _, next_state_values = model(Variable(next_states))
    
    # if this is a final state (lives == 0) then set reward = 0
    indices = next_states.view((batch_size, -1))[:,-1].le(0.1) # select indices of transitions reaching the final state
    next_state_values[indices] = 0.0
    
    """
    print action_scores.exp()
    print states
    print state_values
    """
    R = (next_state_values + Variable(rewards)).detach() # obtained reward
    V = state_values  # expected reward
    
    # compute log_prob of chosen actions
    log_probs = torch.masked_select(action_scores, mask=actions).unsqueeze(1)
    
    L_actor = - log_probs * (R-V).detach()
    L_critic = 1.0 * (R-V)**2
    # L_critic = 0.5 * F.smooth_l1_loss(V, R)
    # L_entropy = 100.0 * (log_probs.exp() * log_probs).sum(dim=-1)
    """
    print L_actor
    print L_critic
    print L_entropy
    """
    
    optimizer.zero_grad()
    loss = (L_actor + L_critic).sum() / batch_size
    # loss = (L_actor + L_critic + L_entropy).sum() / batch_size
    loss.backward()
    optimizer.step()
    
    # sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the network.')
    parser.add_argument('-c', '--clear', action='store_true', help='clear the network before training')
    
    parser.add_argument('-e', '--epsilon', default=0.9, type=float, help='initial probability to choose a random action during training')
    parser.add_argument('-d', '--decay', default=1000.0, type=float, help='epsilon decay')
    parser.add_argument('-f', '--final', default=0.05, type=float, help='final epsilon')
    
    parser.add_argument('-l', '--lrate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('-m', '--memory', default=500, type=int, help='replay memory')
    
    args = parser.parse_args()
    
    if not args.clear and os.path.isfile("model.tar"):
        # load model from file
        model = torch.load("model.tar")
        print >> sys.stderr, "Loaded model from file"
    else:
        model = PolicyNetwork()
        print >> sys.stderr, "Created new model"
    
    optimizer = optim.Adam(model.parameters(), lr=args.lrate)
    memory = ReplayMemory(args.memory)
    
    print "Using learning rate = %r" % args.lrate
    print "Using replay memory = %d" % args.memory
    print
    
    running_avg = 0.0
    running_reward = 0.0
    
    old_reward = None
    old_action = None
    old_state = None
    old_hints = None
    
    try:
        iteration = 0
        while True:
            eps_threshold = args.final + (args.epsilon - args.final) * math.exp(-1. * iteration / args.decay)
            game = Game(num_players=5, ai="extremehanabi", ai_params={'model': model, 'training': eps_threshold}, strategy_log=(random.randint(0,10) == -1))
            
            game.setup()
            old_hints = game.hints
            game_reward = 0.0
            
            # train only on good decks
            if game.deck[0].color == Card.RAINBOW and game.deck[0].number < 5:
                continue
            
            chosen_actions = []
            previous_score = game.get_current_score()
            
            for player, turn in game.run_game():
                score = game.get_current_score()
                lives = game.lives
                reward = 0.0
                
                # game.log_turn(turn, player)
                # game.log_status()
                
                # Hanabi score reward
                reward = score - previous_score
                
                # penalize wrong action
                if model.saved_action.chosen_action[0] == ActionManager.HINT and turn.action.type != Action.HINT and old_hints == 0:
                    reward -= 1.0
                
                reward = float(reward)
                game_reward += reward
                
                previous_score = score
                
                chosen_actions.append(model.saved_action.chosen_action[0])
                
                # store previous transition in memory
                if old_state is not None:
                    memory.push(old_state, old_action, model.saved_action.state, old_reward)
                
                old_reward = torch.Tensor([reward])
                old_state = model.saved_action.state
                old_action = model.saved_action.action_variable
                old_hints = game.hints
                
                # Perform one step of the optimization (on the target network)
                optimize_model(model, optimizer, memory)
                
                # store transition to last turn
                if game.end_game:
                    memory.push(old_state, old_action, torch.zeros_like(old_state), old_reward)
                    optimize_model(model, optimizer, memory)
            
            
            iteration += 1
            
            if iteration % 10 == 0:
                print
                print "Iteration %d" % iteration
                print "Epsilon threshold:", eps_threshold
                print Counter(chosen_actions)
                print game.statistics
                running_avg = 0.99 * running_avg + 0.01 * game.statistics.score
                running_reward = 0.99 * running_reward + 0.01 * game_reward
                print "Running average score: %.2f" % running_avg
                # print "Running average reward: %.2f" % running_reward
                print "Game reward: %.2f" % game_reward
                
                if len(memory) < memory.capacity:
                    print "Memory size:", len(memory)
            
            if iteration % 100 == 0:
                torch.save(model, "model.tar")
            
            # finish_episode(model, optimizer)

    except KeyboardInterrupt:
        print "Interrupted after %d iterations" % iteration
        torch.save(model, "model.tar")

