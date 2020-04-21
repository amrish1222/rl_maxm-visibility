# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.distributions import Categorical
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
import random

torch.manual_seed(10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, env):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                    nn.Conv2d(1,16,(8,8),4,1),
                    nn.ReLU(),
                    nn.Conv2d(16,32,(4,4),2,1),
                    nn.ReLU(),
                    nn.Conv2d(32,32,(3,3),1,1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(6*6*32, 256),
                    nn.ReLU(),
                    nn.Linear(256, len(env.getActionSpace())),
                    nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Conv2d(1,16,8,4,1),
                    nn.ReLU(),
                    nn.Conv2d(16,32,4,2,1),
                    nn.ReLU(),
                    nn.Conv2d(32,32,3,1,1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(6*6*32, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                )

        self.train()
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state.unsqueeze(0))
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class PPO:
    def __init__(self, env):
        self.lr = 0.0002
        self.betas = (0.9, 0.999)
        self.gamma = 0.99 
        self.eps_clip = 0.2
        self.K_epochs = 4
        
        self.policy = ActorCritic(env).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.policy_old = ActorCritic(env).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.sw = SummaryWriter(log_dir=f"tf_log/demo_CNN{random.randint(0, 1000)}")
        print(f"Log Dir: {self.sw.log_dir}")
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
#            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        return advantages.mean()
        
    def formatInput(self, states):
        out = []
#        for state in states:
#            out.append(np.concatenate((state[0], state[1].flatten())))
        for state in states:
            temp = state[1].reshape((1, state[1].shape[0], state[1].shape[1]))
            temp.shape
            out.append(temp)
        return out
    
    def summaryWriter_showNetwork(self, curr_state):
        X = torch.tensor(list(curr_state)).to(self.device)
        self.sw.add_graph(self.model, X, False)
    
    def summaryWriter_addMetrics(self, episode, loss, rewardHistory, mapRwdDict, lenEpisode):
        if loss:
            self.sw.add_scalar('6.Loss', loss, episode)
        self.sw.add_scalar('3.Reward', rewardHistory[-1], episode)
        self.sw.add_scalar('5.Episode Length', lenEpisode, episode)
        
        if len(rewardHistory)>=100:
            avg_reward = rewardHistory[-100:]
            avg_reward = mean(avg_reward)
        else:    
            avg_reward = mean(rewardHistory) 
        self.sw.add_scalar('1.Average of Last 100 episodes', avg_reward, episode)
        
        for item in mapRwdDict:
            title ='4. Map ' + str(item + 1)
            if len(mapRwdDict[item]) >= 100:
                avg_mapReward,avg_newArea, avg_penalty =  zip(*mapRwdDict[item][-100:])
                avg_mapReward,avg_newArea, avg_penalty = mean(avg_mapReward), mean(avg_newArea), mean(avg_penalty)
            else:
                avg_mapReward,avg_newArea, avg_penalty =  zip(*mapRwdDict[item])
                avg_mapReward,avg_newArea, avg_penalty = mean(avg_mapReward), mean(avg_newArea), mean(avg_penalty)

            self.sw.add_scalars(title,{'Total Reward':avg_mapReward,'New Area':avg_newArea,'Penalty': avg_penalty}, len(mapRwdDict[item])-1)
            
    def summaryWriter_close(self):
        self.sw.close()
        
    def saveModel(self, filePath):
        torch.save(self.policy, f"{filePath}/{self.policy.__class__.__name__}.pt")
    
    def loadModel(self, filePath):
        self.model = torch.load(filePath)
        self.model.eval()
