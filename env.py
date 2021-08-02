import time
import os
import pybullet as p
import pybullet_data
import math
import random
from scipy.spatial import distance
import numpy as np


class Env():

    def __init__(self):
        self.physicsClientId = p.connect(p.GUI)  # connect to bullet    
        self.urdfRootPath = pybullet_data.getDataPath()
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0,
                                     cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2])

    def step(self, action):
        self.action = action
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi/2.])
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]

        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(
            self.pandaUid, 11, newPosition, orientation)[0:7]
        p.setJointMotorControlArray(self.pandaUid, list(
            range(7))+[9, 10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])

        p.stepSimulation()

        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        jointStates = np.zeros(7)
        for n in range(7):
            jointStates[n] = p.getJointState(self.pandaUid, n)[0]
        jointStates_ = tuple(jointStates)

        state_fingers = (p.getJointState(self.pandaUid, 9)[
            0], p.getJointState(self.pandaUid, 10)[0])

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

    def reset(self):
        p.resetSimulation()
        # we will enable rendering after we loaded everything
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # urdfRootPath = pybullet_data.getDataPath()
        self.planeUid = p.loadURDF(os.path.join(
            self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
        self.tableUid = p.loadURDF(os.path.join(
            self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
        p.setGravity(0, 0, -10)
        self.pandaUid = p.loadURDF(os.path.join(
            self.urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True)
        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        for j in range(7):
            p.resetJointState(self.pandaUid, j, rest_poses[j])

        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid, 10, 0.08)
        state_object = [random.uniform(
            0.5, 0.8), random.uniform(-0.2, 0.2), 0.05]
        self.objectUid = p.loadURDF(os.path.join(
            self.urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]

        jointStates = np.zeros(7)
        for n in range(7):
            jointStates[n] = p.getJointState(self.pandaUid, n)[0]
        jointStates_ = tuple(jointStates)

        state_fingers = (p.getJointState(self.pandaUid, 9)[
            0], p.getJointState(self.pandaUid, 10)[0])
        observation = state_robot + jointStates_ + state_fingers

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        observation = np.array(observation).astype(np.float32)
        return observation

    def close(self):
        p.disconnect()
