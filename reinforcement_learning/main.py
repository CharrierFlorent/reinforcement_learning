# -*- coding: utf-8 -*-
import gym
from gym import spaces
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import plot
from  RandomAgent import RandomAgent
from QAgent import QAgent
from QLearn import QLearn
#Action : < 0 = pousse gauche, > 0 = pousse droite
#car_pos = -1.2 car_pos = 0.6
#velocite min = -0.07  velocite high = 0.07
#reward = 100 atteindre le drapeau

def show_anim(env, agent, step):
    reward = 0
    observation = env.reset()
    for i in range(step):
        env.render()
        action = [agent.action(observation,reward)]
        observation, reward, done, info = env.step(action) # take a random action
        if done:
            print('success after ', i, ' steps')
            break
    env.close()

def simulation(env, agent, n_episode=20, step = 100, anim=False, plot_stat=False, verbose=False):
    not_random = False
    if hasattr(agent,'ai'):
        not_random =True

    stats = plot.EpisodeStats(episode_lengths=np.zeros(n_episode),episode_rewards=np.zeros(n_episode))   
    avg_reward = 0
    reward = 0
    cumul_reward = []
    decay = 10
    
    for i_episode in range(n_episode):
        observation = env.reset()
        reward_sum = 0
    
        for t in range(step):
            action = [agent.action(observation,reward)]
            observation, reward, done, info = env.step(action)
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            reward_sum += reward
    
            if done:
                if not_random:
                    if agent.ai.epsilon > 0.15 and i_episode%decay == 0:
                      agent.ai.epsilon -= 0.1 
                    break
                else:
                    break
        if verbose:
            print('episode ',i_episode, ' total reward : ', reward_sum, t) 
        avg_reward += stats.episode_rewards[i_episode]
    
    avg_reward /= n_episode
    if verbose:
        print('avg reward : ', avg_reward)
    if anim:
        show_anim(env,agent,step) 
    if plot_stat:
        plot.plot_episode_stats(stats)
    return stats

def test_discretize():
    n_episode = 500
    step = 5000
    states = [4,6,8,10,12,14]
    stats = []
    labels = []
    action_space = np.arange(-1.0,2.0,1.0)
    for nb_states in states :
        print('nb_states ', nb_states)
        env = gym.make('MountainCarContinuous-v0').env
        agent = QAgent(action_space,nb_states)
        stats += [simulation(env,agent, n_episode, step)]
        labels += ['nb states = ' + str(nb_states*nb_states)]
        env.close()
    plot.plot_compare(stats,labels)

def test_random(n_episode, step, action_space, anim, plot_stat, verbose):
    print('random :')
    env = gym.make('MountainCarContinuous-v0').env
    agent = RandomAgent(env.action_space)
    simulation(env,agent, n_episode, step, anim, plot_stat, verbose)
    env.close()

def test_qlearn(n_episode, step, action_space, anim, plot_stat, verbose):
    print('qagent :')
    env = gym.make('MountainCarContinuous-v0').env
    agent = QAgent(action_space,15)
    simulation(env,agent, n_episode, step, anim, plot_stat, verbose)
    env.close()

(anim, plot_stat, verbose) = (True, True, False)
n_episode = 1000
step = 5000
action_space = np.arange(-1.0,2.0,1.0)
#test_discretize()
test_random(n_episode, step, action_space, False, plot_stat, verbose)
test_qlearn(n_episode, step, action_space, anim, plot_stat, verbose)
