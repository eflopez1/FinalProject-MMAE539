# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 20:27:49 2021

@author: elope
"""
import numpy as np
from FastSLAM import particleRobot
import matplotlib.animation as animation
from itertools import product

"""
Defining landmark location
"""


xs = np.arange(-10,11,5)
ys=  np.arange(-10,11,5)
landmarks = []
for p in product(xs,ys):
    landmarks.append(p)
landmarks = np.array(landmarks).T
numLM = max(landmarks.shape)
newCol = np.arange(numLM)
landmarks = np.vstack((landmarks,newCol)).T

robot = particleRobot(initState=[0,0,0],     # Start position of bot (x,y,theta) 
                      sensorRange=1.5,       # How far the sensor can read
                      numParticles=1000,     # Number of particles to estimate the prior
                      alpha=0.001,            # Motion noise
                      sigma = [.1,.1],       # Measurement noise
                      dt=.05,                # Simulation timestep
                      updateDelay = 0.25,     # Time delay between timesteps
                      landmarks = landmarks, # Where the landmarks are in space
                      stealTime = None,         # The time when the bot will be stolen and randomly relocated
                      )          
robot.initWorld()

anim = animation.FuncAnimation(robot.fig, robot.animate, 
                                frames=500,   # Number of frames that will be usedd to make the video
                                interval=100) # Update interval between frames (in ms)

"""
The below will execute the code without animations
"""
# for i in range(1000):
#     robot.run()

vidName = "FastSLAM.mp4"
# Uncomment the below if you want to save the GIF
anim.save(vidName,fps=10)