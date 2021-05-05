# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:21:12 2021

@author: elope
"""

import numpy as np
from numpy import sin, cos, arctan, sqrt, pi, exp, matmul
from numpy.linalg import inv, svd
import matplotlib.pyplot as plt
from ResampleMethods import systematic_resample_augmented as resample
from math import atan2
import matplotlib.collections as clt
from copy import deepcopy # Needed to copy the classes over
from time import time
from scipy.stats import multivariate_normal
multi_pdf = multivariate_normal.pdf
import pdb

class particleRobot:
    """
    Particle robot that follows a velocity motion model
    """
    def __init__(self, initState = [0,0,0], numParticles=1000, dt=.1, 
                 landmarks = None, xLim=(-40,40), yLim=(-20,20), 
                 sensorRange=.5, updateDelay=0.5,
                 alpha = None, sigma=None, stealTime=None):
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
            
        if sigma==None: # Measurement/sensor noise
            self.sigma = np.ones(2)                  # Unit variance, if not given any
        elif type(sigma)==int or type(sigma)==float:
            self.sigma = np.ones(2)*sigma
        else:
            self.sigma = sigma
            
        # This is to see if we are stealing the robot
        self.botSteal = False
        if stealTime !=None:
            self.botSteal = True
            self.stealTime = stealTime
            
            
            
        
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
        self.ellipsePlot, = self.ax.plot([],[], '.', markersize=.05, color='purple', label = 'Covariance Ellipse')
        self.collectionBots = clt.PatchCollection(self.bots)
        self.collectionSensors = clt.PatchCollection(self.sensors)
        self.collectionParticleHeadings = clt.LineCollection(self.particleHeadings)
        
        #### Collection parameters
        botcolor = 'green'
        self.botcolor = botcolor
        self.collectionBots.set_color(botcolor)
        self.collectionSensors.set_edgecolor(botcolor)
        self.collectionSensors.set_facecolor('none')
        self.collectionParticleHeadings.set_color('red')
        self.collectionParticleHeadings.set_linewidths(.25)
        
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
        Note that for fast SLAM, all the particles start at the robot's position
        """
        self.particles = []
        for i in range(self.numParticles):
            particle = Particle(np.copy(self.state))            
            self.particles.append(particle)
        
        self.prevPoses = np.zeros((2,2))
        
        #### Features seen tracked here
        self.featuresSeen = []
                    
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
        
        # Check to make sure new angle is within pi_to_pi!
        thetaP = self.angleCheck(thetaP)
        
        newX = [xP, yP, thetaP]
        return newX
        
        
        
        
        
    def controller(self):
        v = 4    # Moving 1m/sec
        w = pi/4 # Rotating at pi/2 rad/s
        xPrior = self.state
        self.state = self.sample_motion_model_velocity([v,w], xPrior)
        self.prevStates.append(xPrior)  # Only storing after we have left that state.
        
        # Let's also move those particles which represent our aprior
        for i, particle in enumerate(self.particles):
            newPartLoc = self.sample_motion_model_velocity([v,w], particle.getState())
            self.particles[i].updateState(newPartLoc)
        self.time += self.dt
        
        
       
        
        
    def sample_normal_distribution(self, b2):
        """
        Samples a value from a normal distribution
        with variance b^2 and mean 0
        """
        b = sqrt(abs(b2))
        return(0.5*np.sum(np.random.uniform(-b,b,12)))
        



        
    def updateParticles(self):
        """
        The prediction step
        """
        measurements = self.measure()
        
        # Next, if a measurement was in range, then we can update our posterior distribution (i.e. our particles)
        if len(measurements)>0:
            """FastSLAM calculates the weights differently, so they are not needed here."""
            
            print('Measurement Made! Updating EKF.')
            self.EKFUpdate(measurements)
        
               
          
      
          
    def measure(self):
        """ Performs the measurement for the bot and returns a list of measurements taken"""
        
        measurements = []
        print('\nTaking sensor readings')
        measurements = []               # Where we will store measurement readings
        sigmaR, sigmaPhi = self.sigma   # Get measurement noise standard deviations
        botLoc = self.state[:2]         # Get current bot location
        theta = self.state[2]           # Robot heading angle
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
        
        return(measurements)
                
    

            
            
    def EKFUpdate(self, measurements):
        """
        Given the most recent measurements, calculates an EKFUpdate for the particles
        In other words, we are now localizing while mapping an environment.
        Weights are calculated based on the covariance 
        """
        N = len(self.particles)
        weights = np.ones(N)
        
        sigmaR, sigmaPhi = self.sigma
        Q = np.array([[sigmaR**2, 0],
                      [0,sigmaPhi**2]])
        I = np.eye(2)
        
        # First, we must take into account new potential features
        newFeatures = []

        for feature in measurements: # I assume we will not measure many features at once, but just in case we do...
            _,_, lmIndex = feature
            if not lmIndex in self.featuresSeen:
                newFeatures.append(lmIndex)


        for k, particle in enumerate(self.particles):
            x,y,theta = particle.getState() 

            for i, feature in enumerate(measurements):
                r, phi, lmIndex = feature
                z = np.array([r,phi]) #True measurement. Needed later, potentially
                sqrtQ = r # To align with what is in the textbook
                    
                if not lmIndex in self.featuresSeen: # New feature, never seen before:
                    
                    # Calculate landmark position relative to particle
                    muX = x + r*cos(phi + theta)
                    muY = y + r*sin(phi+theta)
                    mu = [muX,muY]
                    
                    H = np.array([[-(muX - x)/sqrtQ, -(muY-y)/sqrtQ],
                                  [(muY-y)/sqrtQ**2, -(muX-x)/sqrtQ**2]])
                    Sigma = matmul(inv(H),matmul(Q,inv(H).T))
                    Sigma = Sigma.tolist()
                    
                    # Adding the mu and sigma directly to the particle
                    particle.addNewLM(mu, Sigma, lmIndex)
                    
                    # Since this is a first time measuring this, the weight is simply the probability of being in this location.
                    weights[k] *= 1 # Random initial weight to all particles
                    
                else:
                    # First, obtain this mean relative to this particle
                    j = self.featuresSeen.index(lmIndex)
                    muOld, Sigma = particle.getEKFinfo(lmIndex)
                    muX, muY = muOld
                    
                    rHat = sqrt((muX-x)**2 + (muY-y)**2)
                    phiHat = atan2(muY-y, muX-x) - theta
                    zHat = np.array([rHat, phiHat]) # Predicted meausrement based on this particle's movement in space
                    zDiff = z-zHat # Will be used to update covariance and mean
                    
                    # Jacobian
                    H = np.array([[-(muX - x)/sqrtQ, -(muY-y)/sqrtQ],
                                  [(muY-y)/sqrtQ**2, -(muX-x)/sqrtQ**2]])
                    
                    # Measurement covariance
                    Qt = matmul(matmul(H,Sigma),H.T) + Q 
                    
                    # Kalman gain
                    K = matmul(matmul(Sigma,H.T),inv(Qt))
                    
                    # Updating mean
                    muNew = muOld + matmul(K,zDiff)
                    muNew = list(muNew)
                    
                    # Updating covariance
                    newSigma = matmul((I - matmul(K,H)),Sigma)

                    # Appending new mu and sigma to particle
                    particle.updateEKFinfo(muNew, newSigma, lmIndex)
                    
                    # Get weight from multivariate pdf
                    w = multi_pdf(zDiff, cov = Qt)
                    weights[k] *= w
                        
        # Normalize weights IFF the weights don't sum to zero
        if np.sum(weights)!=0:
            weights /= np.sum(weights)
        else:
            weights = np.ones(N) # In the event that no weight is prominent, just sample them all the same.
            
        #### Replace old particles with new ones
        indicies = resample(weights)
        newParticles = []
        for loc, replace in enumerate(indicies):
            newParticles.append(deepcopy(self.particles[replace]))
            
        self.particles = newParticles
                        
        # Make sure to add any unseen landmarks into the landmark list!
        if len(newFeatures)>0:
            for featIndex, f in enumerate(newFeatures):
                self.featuresSeen.append(f)
                
                
        

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
        self.ellipsePlot.set_data([],[])  # Plotting covariance ellipses of landmarks
        self.time_text.set_text('')       # For showing time
        
        # Getting new locations and data
        self.controller()         # Move the bot
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
        particleLocs = self.getParticleLocs()
        self.particlePlot.set_data(particleLocs[:,0],particleLocs[:,1])
        self.plotParticleHeadings() # Done in the helper function
        self.plotEllipses()
        self.showState()
        self.collectionBots.set_paths(self.bots)
        self.collectionSensors.set_paths(self.sensors)
        self.time_text.set_text('time = %.1f' % self.time)
        
        
        
    def run(self):
        """
        Use this code to run without matplotlib animations.
        Mostly for debugging purposes.
        """
        
        self.controller()
        if self.measureTime > self.updateDelay:
            self.updateParticles()
        else:
            self.measureTime += self.dt
            
        if self.botSteal and self.time>self.stealTime:
            self.stealBot()
            self.botSteal = False # Will not steal the bot again
    
        self.plotEllipses()
        
        
    def plotParticleHeadings(self):
        """
        A Helper function to plot the headings of the particles
        """
        lines = []
        length = .25
        for particle in self.particles:
            x,y,theta = particle.getState()
            start = x,y
            end = x+length*cos(theta), y+length*sin(theta)
            lines.append([start,end])
        self.collectionParticleHeadings.set_paths(lines)

    
    def getParticleLocs(self):
        particleLocs = np.zeros((self.numParticles,3))
        for i, particle in enumerate(self.particles):
            x,y,theta = particle.getState()
            particleLocs[i] = x,y,theta
    
        return particleLocs
    
    
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
        bot = plt.Polygon(X,color=self.botcolor)
        self.bots.append(bot)
        
        # Now plot the sensor range as a circle around the bot's state
        sensor = plt.Circle([x,y], radius=self.sensorRange, edgecolor=self.botcolor, facecolor=None)
        self.sensors.append(sensor)
        
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
    
    
    def plotEllipses(self):
        covs = [] # Need to average the covariance ellipse 
        mus = []  # Need to average the mean location of each landmark
        xs = []   # List where the x-points will be stored
        ys = []   # List where the y-points will be stored
        lm_indexes = []
        
        if len(self.featuresSeen)>0:
            for feat in self.featuresSeen:
                lm_indexes.append(feat)
                covs.append([])
                mus.append([])
            
            for i, feat in enumerate(self.featuresSeen):
                for particle in self.particles:
                    mu, cov = particle.getEKFinfo(feat)
                    mus[i].append(mu)
                    covs[i].append(cov)
            
            
            mus = np.array(mus)
            covs = np.array(covs)
            
            for j, feat in enumerate(self.featuresSeen):
                cov_mat = np.mean(covs[j], axis=0)
                mu = np.mean(mus[j], axis=0)
                
            
                U, s, rotation = svd(cov_mat)
                radii = 5*sqrt(s[:2])
                # calculate cartesian coordinates for the ellipsoid surface
                u = np.linspace(0.0, 2.0 * np.pi, 400)
                x = radii[0] * np.cos(u)
                y = radii[1] * np.sin(u)
                for i in range(len(x)):
                    [x[i], y[i]] = np.dot([x[i], y[i]], rotation[:2, :2]) + mu
                                
                xs.append(x)
                ys.append(y)
        self.ellipsePlot.set_data(xs,ys)
            
    
class Particle:
    """
    All the particles. Contain information regarding.
    """
    def __init__(self, state):
        x,y,theta = state
        self.x = x
        self.y = y
        self.theta = theta
        self.lm_index = []
        self.lm_pos = []
        self.lm_covariances = []
        
    def updateState(self, newState):
        x, y, theta = newState
        self.x = x
        self.y = y
        self.theta = theta
    
    def getState(self):
        state = [self.x, self.y, self.theta]
        return state
        
    def addNewLM(self, mu, Sigma, lm_index):
        """
        Given a new landmark mean, covariance, and index, add to our current list
        """
        self.lm_index.append(lm_index)
        self.lm_pos.append(mu)
        self.lm_covariances.append(Sigma)
        
    def getEKFinfo(self, lm_index):
        loc = self.lm_index.index(lm_index) # Get the location in the respective lists of the covariacne and mean
        mu = self.lm_pos[loc]
        Sigma = self.lm_covariances[loc]
        
        return(np.array(mu), np.array(Sigma))
    
    def updateEKFinfo(self, mu, Sigma, lm_index):
        """
        Updating a landmark which has previously been seen
        """
        loc = self.lm_index.index(lm_index)
        self.lm_pos[loc] = mu
        self.lm_covariances[loc] = Sigma