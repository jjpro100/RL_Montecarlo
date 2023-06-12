# Spring 2020, Reinforcement Learning
# Monte-Carlo and Temporal-difference policy evaluation

import gym
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D

env = gym.make("Blackjack-v0")

def mc_policy_evaluation(policy, env, num_episodes, gamma=1.0):
    """
        Find the value function for a given policy using first-visit Monte-Carlo sampling
        
        Input Arguments
        ----------
            policy: 
                a function that maps a state to action probabilities
            env:
                an OpenAI gym environment
            num_episodes: int
                the number of episodes to sample
            gamma: float
                the discount factor
        ----------
        
        Output
        ----------
            V: dict (that maps from state -> value)
        ----------
        
    """
    
    # value function
    V = defaultdict(float)
  
    ##### FINISH TODOS HERE #####
    R = defaultdict(float)
    N = defaultdict(float)  
 
    for _ in range(num_episodes):
        s = env.reset()
        S, A, Re = [], [], []
        for n in range(500):
            a = policy(s)
            s_, r, done, _ = env.step(a)  # Retrieve state and reward after taking action
            S.append(s)
            A.append(a)
            Re.append(r)
            if done:
                break
            s = s_
    
        for s in S:
            for n in range(len(S)):
                if S[n] == s:# First ocurrence of state s in the episode
                    n_zero = n
                    break 

            G = 0
            for n, r in enumerate(Re[n_zero:]):
                G += (gamma**n)*r    # Return from first occurence of s

            N[s] += 1.0 
            R[s] += G
            V[s] = R[s]/N[s]  # Update V   
    #############################

    return V


def td0_policy_evaluation(policy, env, num_episodes, gamma=1.0):
    """
        Find the value function for the given policy using TD(0)
    
        Input Arguments
        ----------
            policy: 
                a function that maps a state to action probabilities
            env:
                an OpenAI gym environment
            num_episodes: int
                the number of episodes to sample
            gamma: float
                the discount factor
        ----------
    
        Output
        ----------
            V: dict (that maps from state -> value)
        ----------
        
    """
    # value function
    V = defaultdict(float)

    ##### FINISH TODOS HERE #####
    
    alpha = 0.6
    for _ in range(num_episodes):     
        s = env.reset()
        for n in range(500):
            action = policy(s)
            s_, r, done, _ = env.step(action)
            V[s] = V[s] + alpha * ( r + gamma*V[s_] - V[s] ) # Update rule for TD(0)
            if done:
                break
            s = s_

    

    #############################

    return V

    

def plot_value_function(V, title="Value Function"):

    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))
    
    
def apply_policy(observation):
    """
        A policy under which one will stick if the sum of cards is >= 20 and hit otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


if __name__ == '__main__':
    V_mc_10k = mc_policy_evaluation(apply_policy, env, num_episodes=10000)
    plot_value_function(V_mc_10k, title="10,000 Steps")
    V_mc_500k = mc_policy_evaluation(apply_policy, env, num_episodes=500000)
    plot_value_function(V_mc_500k, title="500,000 Steps")


    V_td0_10k = td0_policy_evaluation(apply_policy, env, num_episodes=10000)
    plot_value_function(V_td0_10k, title="10,000 Steps")
    V_td0_500k = td0_policy_evaluation(apply_policy, env, num_episodes=500000)
    plot_value_function(V_td0_500k, title="500,000 Steps")
    



