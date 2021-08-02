# import gym
import env
import pybullet as p
import numpy as np
import math
from numpy import save
import time


env = env.Env()
env.reset()
done = False
error = 0.01
fingers = 1
info = [0.7, 0, 0.1]

k_p = 10
k_d = 1
dt = 1./240. # the default timestep in pybullet is 240 Hz  
t = 0
dz = 0

for i_episode in range(50):
    observation = env.reset()
    fingers = 1
    for t in range(100):
        # time.sleep(1./100.)
        # print(observation)
        
        dx = info[0]-observation[0]
        dy = info[1]-observation[1]
        target_z = info[2] 
        if abs(dx) < error and abs(dy) < error and abs(dz) < error:
            fingers = 0
        if (observation[3]+observation[4])<error+0.02 and fingers==0:
            target_z = 0.5
        dz = target_z-observation[2]
        pd_x = k_p*dx + k_d*dx/dt
        pd_y = k_p*dy + k_d*dy/dt
        pd_z = k_p*dz + k_d*dz/dt
        action = [pd_x,pd_y,pd_z,fingers]
        
        observation, reward, done, info_ = env.step(action)
        info = list(info_.values())
        info = np.array(info)
        info = info[0]
        print(reward)
        
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()