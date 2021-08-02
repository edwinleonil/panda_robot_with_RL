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


if __name__ == '__main__':
    physicsClientId = p.connect(p.GUI)  # connect to bullet
    # physicsClientId = p.connect(p.DIRECT)  # to launch without GUI
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0,
                                cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2])

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



    def step(action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi/2.])
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]

        currentPose = p.getLinkState(pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(
            pandaUid, 11, newPosition, orientation)[0:7]
        p.setJointMotorControlArray(pandaUid, list(
            range(7))+[9, 10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])

        p.stepSimulation()

        state_object, _ = p.getBasePositionAndOrientation(objectUid)
        state_robot = p.getLinkState(pandaUid, 11)[0]
        jointStates = np.zeros(7)
        for n in range(7):
            jointStates[n] = p.getJointState(pandaUid, n)[0]
        jointStates_ = tuple(jointStates)

        state_fingers = (p.getJointState(pandaUid, 9)[
                         0], p.getJointState(pandaUid, 10)[0])

        dst = distance.euclidean(state_object, state_robot)

        if dst <= 0.01:
            reward = 10
            done = True
        else:
            reward = -(dst)**2
            done = False

        info = {'object_position': state_object}
        observation = state_robot + jointStates_ + state_fingers
        return np.array(observation).astype(np.float32), reward, done, info


    start = time.time()
        


    for i in range(n_games):

        p.resetSimulation()
        # we will enable rendering after we loaded everything
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        urdfRootPath = pybullet_data.getDataPath()
        p.setGravity(0, 0, -10)
        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        planeUid = p.loadURDF(os.path.join(
            urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
        pandaUid = p.loadURDF(os.path.join(
            urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True)
        for j in range(7):
            p.resetJointState(pandaUid, j, rest_poses[j])
        p.resetJointState(pandaUid, 9, 0.08)
        p.resetJointState(pandaUid, 10, 0.08)
        tableUid = p.loadURDF(os.path.join(
            urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
        state_object = [random.uniform(
            0.5, 0.8), random.uniform(-0.2, 0.2), 0.05]
        objectUid = p.loadURDF(os.path.join(
            urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
        state_robot = p.getLinkState(pandaUid, 11)[0]

        jointStates = np.zeros(7)
        for n in range(7):
            jointStates[n] = p.getJointState(pandaUid, n)[0]
        jointStates_ = tuple(jointStates)

        state_fingers = (p.getJointState(pandaUid, 9)[
                         0], p.getJointState(pandaUid, 10)[0])
        observation = state_robot + jointStates_ + state_fingers

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        observation = np.array(observation).astype(np.float32)

        fingers = 1
        

        
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
            observation_, reward, done, info = step(action)
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
