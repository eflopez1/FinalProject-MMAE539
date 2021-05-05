# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:21:12 2021

@author: elope
"""

import numpy as np
from numpy import sin, cos, arctan, sqrt, pi, exp
import matplotlib.pyplot as plt
from ResampleMethods import systematic_resample_augmented as resample
from math import atan2
import matplotlib.collections as clt
from time import time
from scipy.stats import norm
pdf = norm.pdf # Getting the probability density function
import pdb

class particleRobot:
    """
    Particle robot that follows a velocity motion model
    """
    def __init__(self, initState = [0,0,0], numParticles=1000, dt=.1, 
                 landmarks = None, xLim=(-40,40), yLim=(-20,20), 
                 sensorRange=.5, updateDelay=0.5,
                 alpha = None, sigma=None, stealTime=None, augmentedMCL = False, alphaAvg = [0.1,0.7]):
        """
        landmarks: Must be a 2D array where each row is the [x,y,signature] of a landmark
        alpha: An int, float, or array describing the motion noise values for alpha 1-6
        sigma: An int, float, or array describing the measurement noise values for range and bearing.
        """
        self.state = initState # State defined by x,y,theta
        """
        We must ensure the heading angle remains in range -pi<theta<pi!
        """
        self.state[2] = self.angleCheck(self.state[2])
        self.numParticles = numParticles
        self.dt = dt
        self.time = 0        # Saving the time simulation has run
        self.measureTime = 0 # Time since last measurement
        self.sensorRange = sensorRange # in meters.
        self.updateDelay = updateDelay # How many seconds between sensor readings, and hence updates
        
        # Limits of the map
        self.xLim = xLim
        self.yLim = yLim
        
        # Now let's show where these bacons are.
        # Landmarks are designated with [xPos, yPos, signature]
        if type(landmarks) == type(None):
            lm1 = [4,4, 0]
            lm2 = [4,-4, 1]
            lm3 = [-4,-4, 2]
            lm4 = [-4,4, 3]
            self.landmarks = np.array([lm1,lm2,lm3,lm4], np.float)
        else:
            self.landmarks = np.array(landmarks, dtype=np.float)
        
        
        if alpha==None:                              #If the user did not define the motion noise, we will take care of it
            self.alpha = np.ones(6)*0.1              # All motion noise will be a value of 0.1
        elif type(alpha)==int or type(alpha)==float: # A single value for all the motion noise was fed
            self.alpha = np.ones(6)*alpha
        else:                                        # Specific motion values for each variable
            assert len(alpha)==6, "You gave an incorrect number of alphas for motion noise!"
            self.alpha = alpha    
            
        if sigma==None:
            self.sigma = np.ones(2)                  # Unit variance, if not variance fed
        elif type(sigma)==int or type(sigma)==float:
            self.sigma = np.ones(2)*sigma
        else:
            self.sigma = sigma
            
        # This is to see if we are stealing the robot
        self.botSteal = False
        if stealTime !=None:
            self.botSteal = True
            self.stealTime = stealTime
            
        self.augment=False
        if augmentedMCL:
            self.wAvg = np.zeros(2) # [wAvgSlow, wAvgFast]
            self.alphaAvg = alphaAvg
            assert alphaAvg[0]<alphaAvg[1], 'The first coefficient in alpha average must be slower than the second!'
            self.augment=True
            self.setWAvg = True # on the first run, we need to set the moving averages to the current averages of the system
            
            
            
        
    def initWorld(self, figSize=[20,10], dpi=100):
        
        plt.close('all') # If we are starting a world, let's make sure that there is nothign open
        xLim = self.xLim
        yLim = self.yLim
        self.fig = plt.figure(figsize=figSize,dpi=dpi)
        self.ax = self.fig.add_subplot(xlim=xLim, ylim=yLim)
        self.ax.set_xlabel('m')
        self.ax.set_ylabel('m')
        self.ax.set_title('RoboWorld!')
        
        #### Adding empty lists to later store data for plotting
        self.line, = self.ax.plot([],[])  # For bot previous locations
        self.bots = []                    # For bot location
        self.sensors = []                 # For sensor(s)    
        self.particleHeadings = []        # For plotting particle headings
# Uncomment the below if you
        
        #### Initiating the empty collections
        self.particlePlot, = self.ax.plot([],[], 'o', markersize=.2, color='red', label='Particle') # For plotting the particles
        self.collectionBots = clt.PatchCollection(self.bots)
        self.collectionSensors = clt.PatchCollection(self.sensors)
        self.collectionParticleHeadings = clt.LineCollection(self.particleHeadings)
        
        #### Collection parameters
        botColors = ['red','black']
        self.collectionBots.set_color(botColors)
        self.collectionSensors.set_edgecolor('red')
        self.collectionSensors.set_facecolor('none')
        
        #### Adding collections to the axis
        self.ax.add_collection(self.collectionBots)
        self.ax.add_collection(self.collectionSensors)
        self.ax.add_collection(self.collectionParticleHeadings)
        self.time_text = self.ax.text(0.02,0.95,'',transform=self.ax.transAxes) # For showing time
        
        # Create empty list to store previous states
        self.prevStates = [] 
        
        #### Creating particles
        """
        Recall that these particles should be spawned UNIFORMLY initially
        Particle: [x,y,theta] 
        """
        low  = [xLim[0],yLim[0], -pi]
        high = [xLim[1],yLim[1], pi]
        self.particles = np.random.uniform(low,high,(self.numParticles,3))
        for i in range(5):
            self.particles[i] = self.state
        
        #### Plotting landmarks
        self.ax.scatter(self.landmarks[:,0],self.landmarks[:,1], marker='x', color='blue', label='Landmark')
        self.ax.legend(loc='upper right')
        
        
        
        
        
    def sample_motion_model_velocity(self, u, xPrior):
        """
        Motion model of the bot
        
        u: control input compose of [v,w] where..
            v = velocity
            w = rotational velocity
        
        xPriod: The belief of where we are prior
        """        
        v = u[0] # Velocity of bot
        w = u[1] # Angular rotation of bot
        
        # Adding motion noise
        alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = self.alpha
        v += self.sample_normal_distribution(alpha1*v**2 + alpha2*w**2)
        w += self.sample_normal_distribution(alpha3*v**2 + alpha4*w**2)
        gamma = self.sample_normal_distribution(alpha5*v**2 + alpha6*w**2)
        
        if w == 0: r=v
        else: r=v/w
        
        x,y,theta = xPrior
        
        xP  = x - r*sin(theta) + r*sin(theta + w*self.dt)
        yP = y + r*cos(theta) - r*cos(theta + w*self.dt)
        thetaP = theta + w*self.dt + gamma*self.dt
        
        # Check to make sure new angle is good!
        thetaP = self.angleCheck(thetaP)
        
        newX = [xP, yP, thetaP]
        return newX
        
        
        
        
        
    def controller(self):
        v = 5 # Moving 1m/sec
        # w = pi/4 # Rotating at (pi/2) rad/sec]
        w = pi/2
        xPrior = self.state
        self.state = self.sample_motion_model_velocity([v,w], xPrior)
        self.prevStates.append(xPrior)  # Only storing after we have left that state.
        # Let's also move those particles which represent our aprior
        for i, particle in enumerate(self.particles):
            self.particles[i,:] = self.sample_motion_model_velocity([v,w], particle)
        self.time += self.dt
        
        
       
        
        
    def sample_normal_distribution(self, b2):
        """
        Samples a value from a normal distribution
        with variance b^2 and mean 0
        """
        b = sqrt(abs(b2))
        return(0.5*np.sum(np.random.uniform(-b,b,12)))
    
    
    
    
    
    def prob_normal_distribution(self, a, b2):
        """
        Computes a probability density given mean a and variance b2
        """
        return 1/(sqrt(2*pi*b2))*exp(-0.5*(a**2/b2))
      
        
    
    
    
    def landmark_model_known_correspondence(self, z, X, lm):
        """
        Given a measurement z, x, and lm, returns the probability p(z|x,m).
        I.e. returns the probability of having obtained that measurement
        given a known understanding of the map.
        """
        
        # Obtaining parameters
        sigmaR, sigmaPhi= self.sigma # Get measurement noise standard deviation
        r,phi = z[:2]                # Measurement range, bearing angle (don't need index)
        x,y,theta = X                # Bot/Particle position (x,y) and heading angle (theta)
        mx, my, sig = lm             # Landmark position (x,y) and signature (index)
        
        # Belief of where the particle/bot should be relative to landmark
        rHat = sqrt((mx-x)**2+ (my-y)**2)
        phiHat = atan2(my-y,mx-x) - theta
        
        # Get probabilities
        # q1 = self.prob_normal_distribution(r-rHat,sigmaR**2)
        # q2 = self.prob_normal_distribution(phi-phiHat,sigmaPhi**2) 
        
        # Try using the external probability density function
        q1 = pdf(r-rHat, scale=sigmaR)
        q2 = pdf(phi-phiHat, scale=sigmaPhi)
        q3 = 1 # Assume perfect correspondence.
        return(q1*q2*q3)
        
        
        
        
    def updateParticles(self):
        """
        The prediction step
        """
        print('Taking sensor readings')
        measurements = []               # Where we will store measurement readings
        lmFound = []
        sigmaR, sigmaPhi = self.sigma   # Get measurement noise standard deviations
        botLoc = self.state[:2]         # Get current bot location
        theta = self.state[2]
        x,y = botLoc                    # (x,y) coodrinates of bot location
        numLM, _ = self.landmarks.shape # How many landmarks are there?
        
        # Iterate through all the landmarks
        for i in range(numLM):
            lmLoc = self.landmarks[i,:2] # Location of the current landmark
            dis = np.linalg.norm(botLoc - lmLoc)
            
            # First we need measurement readings w.r.t. each landmark
            if dis<=self.sensorRange: #Landmark in sensorrange of bot
                """
                The bot is within range of a landmark
                We must now construct a measurement z
                """
                r = dis
                mx = self.landmarks[i,0]
                my = self.landmarks[i,1]
                phi = atan2(my - y, mx-x) - theta
                
                # Adding some measurement noise 
                r += self.sample_normal_distribution(sigmaR**2)
                phi += self.sample_normal_distribution(sigmaPhi**2)
                
                measurements.append([r,phi,i]) # Appending the landmark range, bearing, and index

        # Next, if a measurement was in range, then we can update our posterior distribution (i.e. our particles)
        if len(measurements)>0:
            # Initialize the weights
            N = len(self.particles)
            weights = np.ones(N)
            
            # Iterate through all the measurements (if more than one taken) and particles
            for z in measurements:
                lm = self.landmarks[z[-1]] # Get the landmark that was measured
                for index, particle in enumerate(self.particles):
                    # Note that we do a times-equals, as if there are multiple landmark readings we want to add the probability from each measurement
                    weights[index] *= self.landmark_model_known_correspondence(z, particle, lm)
                        
            # Normalize weights IFF the weights don't sum to zero
            if np.sum(weights)!=0:
                weights /= np.sum(weights)
            else:
                weights = np.ones(N) # In the event that no weight is prominent, just sample them all the same.
            
            # If we are performing the augmented MCL, then we must sample from the
            if self.augment:
                self.augmentMCL(weights, lm)
            
            else: # If not using augmented, then perform the regular update
                self.standardMCL(weights) # Pass the augmented MCL the most recent landmark measured
                
                
                
                
                
    def standardMCL(self, weights):
        # Perform resampling using resampling algorithm
        indicies = resample(weights)
            
        #Finally, update which particles stay
        newParticles = np.zeros(self.particles.shape)
        for i,particleIndex in enumerate(indicies):
            newParticles[i] = self.particles[particleIndex]
                
        self.particles = newParticles
                
        
        
        
        
    def augmentMCL(self, weights, lm):
        """
        Given the average weights in the last step, computes the moving average
        and adds random particles if needed
        """
        wAvg = np.mean(weights)
        wAvg = np.ones(2)*wAvg
        
        # Try, on the first measurement, to set wAvg as the average of the weights
        if self.setWAvg:
            print('Setting wAvg')
            self.wAvg = wAvg
            self.setWAvg=False
        else:
            self.wAvg += self.alphaAvg*(wAvg - self.wAvg) # Updating the fast and slow averages of weights
        
        try:
            ratio=self.wAvg[1]/self.wAvg[0]
        except:
            ratio = 1
            
        print('Moving Average Ratio:', ratio)
        
        # If the ratio is equal to one, then we will not be adding any random 
        # particles. Also applies if ratio is 
        if ratio==1 or (1-ratio)<0:
            self.standardMCL(weights)
        else:
            print('Using Augmented MCL!!\n')
            # Gathering last landmark read and putting bounds on new random 
            # particles to be in the vacinity
            x,y,_ = lm
            low = [x-3,y-3,-pi]
            high = [x+3,y+3,pi]
            
            # new particles go here:
            newParticles = np.zeros(self.particles.shape)
            
            #Calculating the probability of selecting a random particle
            N = len(weights)
            prob = 1 - ratio
            numsForProb = np.random.uniform(size=N)
            
            # Collecting new particles
            numRandomAdded = 0
            for i in range(N):
                if numsForProb[i]<=prob:
                    newParticles[i] = np.random.uniform(low,high)
                    numRandomAdded+=1
                else:
                    index = resample(weights,1)
                    newParticles[i] = self.particles[index]
                    
            if numRandomAdded>100:
                """
                If too many new particles were added, then let's cut back on the average
                """
                print('Setting wAvg')
                self.wAvg = wAvg
            self.particles = newParticles
            

    def stealBot(self):
        """ A function to randomly relocate the bot """
        xLim = self.xLim
        yLim = self.yLim
        
        low  = [xLim[0],yLim[0], -pi]
        high = [xLim[1],yLim[1], pi]
        # self.state = np.random.uniform(low,high)
        
        self.state = [-7,-7,0]


    def animate(self, i=None):
        # Clearing what we previously had
        self.bots = []                    # For plotting bot
        self.sensors = []                 # For plotting sensors
        self.particleHeadings = []        # Fpr plotting the particle headings
        self.line.set_data([],[])         # For plotting previous positions of bot
        self.particlePlot.set_data([],[]) # Plotting particles
        self.time_text.set_text('')       # For showing time
        
        # Getting new locations and data
        self.controller()         # Move the bot
        self.showState()          # Plotting the bot and particles where they are NOW
        prevStates = np.asarray(self.prevStates)
        
        # Prediction update
        if self.measureTime > self.updateDelay:
            self.updateParticles()
            self.measureTime = 0
        else:
            self.measureTime += self.dt
        
        if self.botSteal and self.time>self.stealTime:
            self.stealBot()
            self.botSteal = False # Will not steal the bot again
        
        # Adding new locations and data to animation
        self.line.set_data(prevStates[:,0],prevStates[:,1])
        self.particlePlot.set_data(self.particles[:,0],self.particles[:,1])
        self.plotParticleHeadings() # Done in the helper function
        self.collectionBots.set_paths(self.bots)
        self.collectionSensors.set_paths(self.sensors)
        self.time_text.set_text('time = %.1f' % self.time)        
        
        
    def run(self):
        """
        Use this code to run without matplotlib animations.
        """
        
        self.controller()
        if self.measureTime > self.updateDelay:
            self.updateParticles()
        else:
            self.measureTime += self.dt
            
        if self.botSteal and self.time>self.stealTime:
            self.stealBot()
            self.botSteal = False # Will not steal the bot again
        
        
    def plotParticleHeadings(self):
        """
        A Helper function to plot the headings of the particles
        """
        lines = []
        length = .25
        for particle in self.particles:
            x,y,theta = particle
            start = x,y
            end = x+length*cos(theta), y+length*sin(theta)
            lines.append([start,end])
        self.collectionParticleHeadings.set_paths(lines)
        self.collectionParticleHeadings.set_color('red')
        self.collectionParticleHeadings.set_linewidths(.25)
    
    
    
    
    
    def showState(self):
        """
        Given some plot (or whatever), show where the bot is now
        The bot position is marked using an isoceles triangle
        """
        a =  1                    # This is the length of the longer side of the isoceles traingle
        beta = arctan(sqrt(15)/3) # Calculated analytically. Don't worry about it.
        l = a*sqrt(15)/12         # The height of the triangle divided by 3
        d = sqrt((a**2)/6)        # Length from center of triangle to lower points
        
        x,y,theta = self.state
        
        # Top point (joining of longest sides)
        Ax = x + 2*l*cos(theta)
        Ay = y + 2*l*sin(theta)
        A = [Ax,Ay]
        
        # Bottom Left point
        Bx = x + d*cos(theta+(pi/2)+beta)
        By = y + d*sin(theta+(pi/2)+beta)
        B = [Bx,By]
        
        # Bottom right point
        Cx = x + d*cos(theta+(3*pi/2)-beta)
        Cy = y + d*sin(theta+(3*pi/2)-beta)
        C = [Cx,Cy]
        
        X = np.array([A,B,C])
        bot = plt.Polygon(X)
        self.bots.append(bot)
        
        # Now plot the sensor range as a circle around the bot's state
        sensor = plt.Circle([x,y], radius=self.sensorRange, edgecolor='red', facecolor=None)
        self.sensors.append(sensor)
        
        # Also, get the average particle location to show where the system thinks it is
        meanState = np.mean(self.particles,axis=0)
        x,y,theta = meanState
        # Top point (joining of longest sides)
        Ax = x + 2*l*cos(theta)
        Ay = y + 2*l*sin(theta)
        A = [Ax,Ay]
        
        # Bottom Left point
        Bx = x + d*cos(theta+(pi/2)+beta)
        By = y + d*sin(theta+(pi/2)+beta)
        B = [Bx,By]
        
        # Bottom right point
        Cx = x + d*cos(theta+(3*pi/2)-beta)
        Cy = y + d*sin(theta+(3*pi/2)-beta)
        C = [Cx,Cy]
        
        X = np.array([A,B,C])
        bot = plt.Polygon(X)
        self.bots.append(bot)
            
    def angleCheck(self, angle):
        """
        Checks to ensure the angle is between -pi and pi. If it is, then it gets passed.
        Otherwise, it gets converted
        """
        if angle>=pi or angle<-pi:
            angle=self.pi_2_pi(angle)
        else:
            pass
        return angle
        
        
    def pi_2_pi(self, angle):
        """
        Needed to ensure the heading angles are consistent
        """
        return (angle + pi) % (2 * pi) - pi