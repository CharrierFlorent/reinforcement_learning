# -*- coding: utf-8 -*-
import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt


EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])


def plot_episode_length(stats):
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Duree des episodes au cours du temps")
    plt.show(fig1)
    return fig1


def plot_reward_steps(stats, smoothing_window=10):
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward en fonction du temps")
    plt.show(fig2)
    return fig2

def plot_episode_complete(stats):
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Steps")
    plt.ylabel("Episode")
    plt.title("Episode par time step")
    plt.show(fig3)
    return fig3

def plot_compare(stats_list, labels, smoothing_window=10):
    fig2 = plt.figure(figsize=(10,5))
    i = 0
    for stats in stats_list:
        rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed,label=labels[i])
        i += 1
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward en fonction du temps")
    plt.legend()
    plt.show(fig2)

    fig1 = plt.figure(figsize=(10,5))
    i = 0
    for stats in stats_list:
        plt.plot(stats.episode_lengths,label=labels[i])
        i += 1
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Duree des episodes au cours du temps")
    plt.legend()
    plt.show(fig1)
    

def plot_episode_stats(stats, smoothing_window=10):
    fig1 = plot_episode_length(stats)
    fig2 = plot_reward_steps(stats)
    fig3 = plot_episode_complete(stats)
    return fig1, fig2, fig3