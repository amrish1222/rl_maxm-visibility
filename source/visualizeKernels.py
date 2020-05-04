# -*- coding: utf-8 -*-


from ppo_agent import PPO
from ppo_agent import Memory
from tqdm import tqdm
from env import Env
import cv2
import torch
import matplotlib.pyplot as plt

env = Env()
rlAgent = PPO(env)

memory = Memory()
rlAgent.loadModel('checkpoints/ppoAdv3P100.pt')

rlAgent.getKernels()


def getLayerOutput(model, img):
    output = []
    fig = plt.figure(3, figsize=(64,64))
    fig.patch.set_facecolor('xkcd:mint green')
    ax1 = fig.add_subplot(8,16,1)
    outs = img/255
    outs = img.squeeze().cpu().detach().numpy()
    ax1.imshow(outs, cmap = 'gray')
    ax1.axis('off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    
    row = 0
    for name, layer in model._modules.items():
        print(name)
#        output.append(layer(img).squeeze().cpu().detach().numpy())
        img = layer(img)
        outs = img.squeeze().cpu().detach().numpy()
        if name in ['1', '3', '5']:
            row += 32
            for i in range(outs.shape[0]):
                    ax1 = fig.add_subplot(8,16,row+ i+1)
                    ax1.imshow(outs[i], cmap = 'gray')
                    ax1.axis('off')
                    ax1.set_xticklabels([])
                    ax1.set_yticklabels([])
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show(block = False)
        plt.show()
        plt.pause(0.005)
    return output

curRawState = env.reset()
curState = rlAgent.formatInput(curRawState)

for step in range(500):
   
    # Get agent actions
    aActions = []
    for i in range(1):
        # get action for each agent using its current state from the network
        action = rlAgent.policy_old.act(curState[i], memory)
        aActions.append(action)
    
    # do actions
    agentPosList, advrsyPosList, display, reward, newAreaVis, penalty, done = env.step(aActions)
    
    # update nextState
    newRawState = []
    for agentPos in agentPosList:
        newRawState.append([agentPos, advrsyPosList, display])
    newState = rlAgent.formatInput(newRawState)
    
    state1 = torch.from_numpy(newState[0][0]).float().to(rlAgent.device).unsqueeze(0)
   
    getLayerOutput(rlAgent.policy.feature1, state1)
    
    break