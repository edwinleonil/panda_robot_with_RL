import time
import os
import pybullet as p
import pybullet_data
import math
import random
from scipy.spatial import distance
import numpy as np
from ddpg_agent import Agent
import matplotlib.pyplot as plt
from env import Env

if __name__ == '__main__':
    env = Env()
    # physicsClientId = p.connect(p.GUI)  # connect to bullet
    # physicsClientId = p.connect(p.DIRECT)  # to launch without GUI
    # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0,
    #                             cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2])

    agent = Agent(alpha=1e-4,
                  beta=1e-3,
                  tau=1e-3,
                  gamma=0.99,
                  input_dims=(12,),
                  n_actions=4,
                  max_size=int(1e5),
                  batch_size=128,
                  fc1_dims=400,
                  fc2_dims=300,
                  action_limit=1)  # define max and min limit for action output

    n_games = 10  # number of episodes for training
    filename = 'Panda_' + str(agent.alpha) + '_beta_' + \
               str(agent.beta) + '_' + str(n_games) + '_games'

    figure_file = 'plots/' + filename + '.png'

    best_score = -500
    score_history = []

    start = time.time()
        


    for i in range(n_games):

        observation = env.reset()  # get observations
        

        
        done = False
        score = 0
        agent.noise.reset()  # Initialize a random process N for action exploration
        iterations = 500
        for t in range(iterations):
            # while not done:
            p.stepSimulation()
            # Select action accordin to current policy and exploration noise
            action = agent.choose_action(observation)
            # Execute action
            observation_, reward, done, info = env.step(action)
            if t == iterations:
                done = True
            # Store transition
            agent.remember(observation, action, reward, observation_, done)
           
            # Sample a random minibatch and update networks
            agent.learn()

            observation = observation_

            score += reward
            if done:
                break

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)

    end = time.time()
    print(f"Runtime of the program is {end - start}")
    x = [i+1 for i in range(n_games)]
    # plot_learning_curve(x, score_history, figure_file)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x, score_history)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    p.disconnect()
