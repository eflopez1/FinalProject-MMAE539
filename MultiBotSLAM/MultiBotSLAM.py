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
                 landmarks = None, xLim=(-50,20), yLim=(-20,20), 
                 sensorRange=.5, updateDelay=0.5,
                 alpha = None, sigma=None, bot2Start = [-30,0,0]):
        """
        landmarks: Must be a 2D array where each row is the [x,y,signature] of a landmark
        alpha: An int, float, or array describing the motion noise values for alpha 1-6
        sigma: An int, float, or array describing the measurement noise values for range and bearing.
        """
        self.state1 = initState # State defined by x,y,theta
        """
        We must ensure the heading angle remains in range -pi<theta<pi!
        """
        self.state1[2] = self.angleCheck(self.state1[2])
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
        
        #### Initialize Bot 2
        self.state2 = bot2Start
        self.state2[2] = self.angleCheck(self.state2[2])
        
        #### Storing start state
        self.initState1 = self.state1 # The first state for bot 1
        self.initState2 = self.state2 # The first state for bot 2
        self.botsMet = False # We only care about the first time that the bots have met
        
        #### Controller updates
        self.goStraight = False
        self.circlesAgain = False
        
        
            
            
    def initWorld(self, figSize=[20,10], dpi=100):
        
        plt.close('all') # If we are starting a world, let's make sure that there is nothign open
        xLim = self.xLim
        yLim = self.yLim
        self.fig = plt.figure(figsize=figSize,dpi=dpi, constrained_layout=True)
        gs = self.fig.add_gridspec(2,2)
        self.ax = self.fig.add_subplot(gs[0,:],xlim=xLim, ylim=yLim)
        self.ax.set_xlabel('m')
        self.ax.set_ylabel('m')
        self.ax.set_title('RoboWorld!')
        
        #### Adding empty lists to later store data for plotting
        self.line1, = self.ax.plot([],[])  # For bot 1 previous locations
        self.line2, = self.ax.plot([],[])  # For bot 2 previous locations
        self.bots = []                    # For bot location
        self.sensors = []                 # For sensor(s)    
        self.particleHeadings1 = []        # For plotting particle headings
        self.particleHeadings2 = []       # For storing particle headings of bot 2
        
        #### Initiating the empty collections
        self.particlePlot1, = self.ax.plot([],[], 'o', markersize=.2, color='red', label='Particle Bot i') # For plotting the particles of bot1
        self.particlePlot2, = self.ax.plot([],[],'o',markersize=.2,color='saddlebrown',label='Particle Bot j')   # For plotting the particles of bot2
        self.ellipsePlot, = self.ax.plot([],[], '.', markersize=.05, color='purple', label = 'Covariance Ellipse')
        self.collectionBots = clt.PatchCollection(self.bots)
        self.collectionSensors = clt.PatchCollection(self.sensors)
        self.collectionParticleHeadings1 = clt.LineCollection(self.particleHeadings1)
        self.collectionParticleHeadings2 = clt.LineCollection(self.particleHeadings2) 
        
        #### Collection parameters
        botcolor = ['green','orangered']
        self.collectionBots.set_color(botcolor)
        self.collectionSensors.set_edgecolor(botcolor)
        self.collectionSensors.set_facecolor('none')
        self.collectionParticleHeadings1.set_color('red')
        self.collectionParticleHeadings1.set_linewidths(.25)
        self.collectionParticleHeadings2.set_color('saddlebrown')
        self.collectionParticleHeadings2.set_linewidths(.25)
        
        #### Adding collections to the axis
        self.ax.add_collection(self.collectionBots)
        self.ax.add_collection(self.collectionSensors)
        self.ax.add_collection(self.collectionParticleHeadings1)
        self.ax.add_collection(self.collectionParticleHeadings2) 
        self.time_text = self.ax.text(0.02,0.95,'',transform=self.ax.transAxes) # For showing time
        
        #### Plotting landmarks
        self.ax.scatter(self.landmarks[:,0],self.landmarks[:,1], marker='x', color='blue', label='Landmark')
        
        # Create empty list to store previous states
        self.prevStates1 = [] # Bot 1
        self.prevStates2 = [] # Bot 2
                
        #### Adding sub bot plot to show where they think they are
        # Bot 1
        self.subAx1 = self.fig.add_subplot(gs[1,1])
        self.subAx1.set_title('Robot i Map')
        self.lineAx1, = self.subAx1.plot([],[]) # Robot mean path
        self.foundLandmarks1, = self.subAx1.plot([],[],'o',label='Found Landmarks')
        # Bot 1 robot representation (i.e. the triangle)
        self.bot1 = []
        self.collectionBot1 = clt.PatchCollection(self.bot1)
        self.subAx1.add_collection(self.collectionBot1)
        # Define some plot properties
        self.subAx1.set_xlabel('m')
        self.subAx1.set_ylabel('m')
        self.subAx1.legend(loc='upper right')
        self.subAx1.set_xlim(xmin=-20,xmax=20)
        self.subAx1.set_ylim(ymin=-20,ymax=20)
        
        # Bot 2
        self.subAx2 = self.fig.add_subplot(gs[1,0])
        self.subAx2.set_title('Robot j Map')
        self.lineAx2, = self.subAx2.plot([],[]) # Robot mean path
        self.foundLandmarks2, = self.subAx2.plot([],[],'o',label='Found Landmarks')
        # Bot 2 robot representation (i.e. the triangle)
        self.bot2 = []
        self.collectionBot2 = clt.PatchCollection(self.bot2)
        self.subAx2.add_collection(self.collectionBot2)   
        # Define some plot properties
        self.subAx2.set_xlabel('m')
        self.subAx2.set_ylabel('m')
        self.subAx2.legend(loc='upper right')
        self.subAx2.set_xlim(xmin=-20,xmax=20)
        self.subAx2.set_ylim(ymin=-20,ymax=20)
            
        #### Creating particles
        """
        Recall that these particles should be spawned UNIFORMLY initially
        Particle: [x,y,theta] 
        Note that for fast SLAM, all the particles start at the robot's position
        """
        self.particles1 = []  # Particles for bot 1
        self.particles2 = [] # Particles for bot 2
        for i in range(self.numParticles):
            particle1 = Particle(np.copy(self.state1))
            particle2 = Particle(np.copy(self.state2))            
            self.particles1.append(particle1)
            self.particles2.append(particle2)
        
        #### Features seen tracked here
        self.featuresSeen1 = [] # Bot1
        self.featuresSeen2 = [] # Bot2
        
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
        
        # Two conditions must be met to go straight;
        # (1) We are past the initial time (say past 2 seconds)
        # (2) The robot's are facing straight
        _,_, currentHead1 = self.state1
        x2,_, currentHead2 = self.state2
        if self.time >2 and -1e-1<currentHead2<1e-1:
            self.goStraight = True
            
        if self.time>10 and x2>0:
            self.circlesAgain = True
        if self.goStraight:
            if 0<currentHead1<pi:
                w=1e-1
            else:
                w=-1e-1
        if self.circlesAgain:
            w=-pi/4
        
        #Bot 1
        xPrior1 = self.state1
        self.state1 = self.sample_motion_model_velocity([v,w], xPrior1)
        self.prevStates1.append(xPrior1)  # Only storing after we have left that state.
        
        if self.goStraight:
            if currentHead2<0:    
                w=1e-2
            else:
                w=-1e-2
        if self.circlesAgain:
            w=-pi/4
        #Bot 2
        xPrior2 = self.state2
        self.state2 = self.sample_motion_model_velocity([v,w], xPrior2)
        self.prevStates2.append(xPrior2)
        
        # Let's also move those particles which represent our aprior
        for i in range(self.numParticles):
            
            # Get the particles
            particle1 = self.particles1[i]
            particle2 = self.particles2[i]
            
            # New particle locations
            newPartLoc1 = self.sample_motion_model_velocity([v,w], particle1.getState())
            newPartLoc2 = self.sample_motion_model_velocity([v,w], particle2.getState())
            
            # Updating particle locations
            self.particles1[i].updateState(newPartLoc1)
            self.particles2[i].updateState(newPartLoc2)
            
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
        
        self.measureBotsRelPos()
        
        # Next, if a measurement was in range, then we can update our posterior distribution (i.e. our particles)
        for botIndex, meas in enumerate(measurements):
            
            if len(meas)>0:
                """FastSLAM calculates the weights differently, so they are not needed here."""
                self.EKFUpdate(meas, botIndex)
        
               
          
      
          
    def measure(self):
        """ Performs the measurement for the bot and returns a list of measurements taken"""
        

        print('\nTaking sensor readings')
        measurements = [[],[]]               # Storing measurements for both bots 1 and 2
        states = [self.state1, self.state2]
        sigmaR, sigmaPhi = self.sigma   # Get measurement noise standard deviations
        numLM, _ = self.landmarks.shape # How many landmarks are there?
        
        for j in range(2):
            
            botLoc = states[j][:2]         # Get current bot location
            theta = states[j][2]           # Robot heading angle
            x,y = botLoc                    # (x,y) coodrinates of bot location
            
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
                    
                    measurements[j].append([r,phi,i]) # Appending the landmark range, bearing, and index
        
        return(measurements)
                
    

            
            
    def EKFUpdate(self, measurements, botIndex):
        """
        Given the most recent measurements, calculates an EKFUpdate for the particles
        In other words, we are now localizing while mapping an environment.
        Weights are calculated based on the covariance 
        """
        
        """
        I am stupid and set the code up a context-specific way. This can be 
        remedied in the future, but for the sake of the project timeline, 
        I will not worry about it right now.
        """
        
        if botIndex==0:
            featuresSeen = self.featuresSeen1
            particles = self.particles1
        
        elif botIndex==1:
            featuresSeen = self.featuresSeen2
            particles = self.particles2
        
        N = self.numParticles
        weights = np.ones(N)
        
        sigmaR, sigmaPhi = self.sigma
        Q = np.array([[sigmaR**2, 0],
                      [0,sigmaPhi**2]])
        I = np.eye(2)
        
        # First, we must take into account new potential features
        newFeatures = []

        for feature in measurements: # I assume we will not measure many features at once, but just in case we do...
            _,_, lmIndex = feature
            if not lmIndex in featuresSeen:
                newFeatures.append(lmIndex)


        for k, particle in enumerate(particles):
            x,y,theta = particle.getState() 

            for i, feature in enumerate(measurements):
                r, phi, lmIndex = feature
                z = np.array([r,phi]) #True measurement. Needed later, potentially
                sqrtQ = r # To align with what is in the textbook
                    
                if not lmIndex in featuresSeen: # New feature, never seen before:
                    
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
                    j = featuresSeen.index(lmIndex)
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
            newParticles.append(deepcopy(particles[replace]))
            
        if botIndex==0:
            # Make sure to add any unseen landmarks into the landmark list!
            self.particles1 = newParticles
            if len(newFeatures)>0:
                for featIndex, f in enumerate(newFeatures):
                    self.featuresSeen1.append(f)
        
        elif botIndex==1:
            self.particles2 = newParticles
            if len(newFeatures)>0:
                for featIndex, f in enumerate(newFeatures):
                    self.featuresSeen2.append(f)
                    
                
    def measureBotsRelPos(self):
        """
        Given the two bots' current location, takes a measurement of the 
        poses of the two bots relative to each other if they are within range.
        """
        botLoc1 = np.array(self.state1[:2])
        botLoc2 = np.array(self.state2[:2])
        
        dis = np.linalg.norm(botLoc1 - botLoc2)
    
        if (dis<=self.sensorRange) and (self.botsMet!=True):
            print('\n--------\nBots are in range!\nTime to update map!\n---------')
            """
            Note: A lot of this notation comes from the 2011 paper by Carlone et. al
            """
            
            # First, change the relative position of bot 2, which we consider 
            # bot 'j'. It should now have the same init state as bot 1
            self.initState2 = self.initState1
            
            # Next, change the limits of BOTH the plots to make space for the 
            # new information
            xLow =-20
            xHigh = 40
            self.subAx1.set_xlim(xmin=xLow, xmax=xHigh)
            self.subAx2.set_xlim(xmin=xLow, xmax=xHigh)
            
            # Next, we must go through the particles of each bot, extract the 
            # mus and sigmas that they have collected, and add them to the 
            # particles of the other bot. We will do this by calculating the mean
            # mu and Sigma for each seen landmark and sharing them equally 
            # with the particles in the other set
            
            # Adding bot 1 information to bot 2
            covs1 = []
            mus1 = []
            lmCorrespond1 = []
            for feat in self.featuresSeen1:
                self.featuresSeen2.append(feat)
                mus1.append([])
                covs1.append([])
                
            for particle in self.particles1:
                for index, j in enumerate(self.featuresSeen1):
                    mu, Sigma = particle.getEKFinfo(j)
                    mus1[index].append(mu)
                    covs1[index].append(Sigma)
                    lmCorrespond1.append(j)
            mus1 = np.array(mus1)
            covs1 = np.array(covs1)
            
            for j, feat in enumerate(self.featuresSeen1):
                lmLoc = np.mean(mus1[j],axis=0)
                cov = np.mean(covs1[j],axis=0)
                featIndex = lmCorrespond1[j]
                
                for particle in self.particles2:
                    particle.addNewLM(lmLoc.tolist(),cov,featIndex)
            
            # Adding bot 2 info to bot 1
            covs2 = []
            mus2 = []
            lmCorrespond2 = []
            for feat in self.featuresSeen2:
                self.featuresSeen1.append(feat)
                mus2.append([])
                covs2.append([])
                
            for particle in self.particles2:
                for index,j in enumerate(self.featuresSeen2):
                    mu, Sigma = particle.getEKFinfo(j)
                    mus2[index].append(mu)
                    covs2[index].append(Sigma)
                    lmCorrespond2.append(j)
            
            mus2 = np.array(mus2)
            covs2 = np.array(covs2)
            
            for j, feat in enumerate(self.featuresSeen2):
                lmLoc = np.mean(mus2[j],axis=0)
                cov = np.mean(covs2[j],axis=0)
                featIndex = lmCorrespond2[j]
                
                for particle in self.particles1:
                    particle.addNewLM(lmLoc.tolist(),cov,featIndex)
            
            del covs1, mus1, lmCorrespond1, covs2, mus2, lmCorrespond2
            
            
            # Finally, denote that the bots have met, so we do not do this process again
            self.botsMet=True
            
            
            

    def animate(self, i=None):
        # Clearing what we previously had
        self.bots = []                     # For plotting bot
        self.sensors = []                  # For plotting sensors
        self.particleHeadings1 = []        # For plotting bot1 particle headings
        self.particleHeadings2 = []        # For plotting bot2 particle headings
        self.line1.set_data([],[])         # For plotting previous positions of bot 1
        self.line2.set_data([],[])         # For plotting previous positions of bot 2
        self.particlePlot1.set_data([],[]) # Plotting bot1 particles
        self.particlePlot2.set_data([],[]) # Plotting bot2 particles
        self.ellipsePlot.set_data([],[])   # Plotting covariance ellipses of landmarks
        self.time_text.set_text('')        # For showing time
        
        # Getting new locations and data
        self.controller()         # Move the bot
        prevStates1 = np.asarray(self.prevStates1)
        prevStates2 = np.asarray(self.prevStates2)
        
        # Prediction update
        if self.measureTime > self.updateDelay:
            self.updateParticles()
            self.measureTime = 0
        else:
            self.measureTime += self.dt
        
        # Adding new locations and data to animation
        self.line1.set_data(prevStates1[:,0],prevStates1[:,1])
        self.line2.set_data(prevStates2[:,0],prevStates2[:,1])
        particleLocs1, particleLocs2 = self.getParticleLocs()
        self.particlePlot1.set_data(particleLocs1[:,0],particleLocs1[:,1])
        self.particlePlot2.set_data(particleLocs2[:,0], particleLocs2[:,1])
        self.plotParticleHeadings() # Done in the helper function
        self.plotEllipses()
        self.showState()
        self.collectionBots.set_paths(self.bots)
        self.collectionSensors.set_paths(self.sensors)
        self.time_text.set_text('time = %.1f' % self.time)
        
        #### Plot subplot information
        self.plotSubBots()
        
        
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
            
        self.plotEllipses()
        
        
    def plotParticleHeadings(self):
        """
        A Helper function to plot the headings of the particles
        """
        lines1 = [] # Particles bot1
        lines2 = [] # Particles bot2
        length = .25
        for i in range(self.numParticles):
            # Bot 1
            x1,y1,theta1 = self.particles1[i].getState()
            start1 = x1,y1
            end1 = x1+length*cos(theta1), y1+length*sin(theta1)
            lines1.append([start1,end1])
            
            # Bot2
            x2,y2,theta2 = self.particles2[i].getState()
            start2 = x2,y2
            end2 = x2+length*cos(theta2), y2+length*sin(theta2)
            lines2.append([start2,end2])
            
        self.collectionParticleHeadings1.set_paths(lines1)
        self.collectionParticleHeadings2.set_paths(lines2)

    
    def getParticleLocs(self):
        particleLocs1 = np.zeros((self.numParticles,3))
        particleLocs2 = np.zeros((self.numParticles,3))
        for i in range(self.numParticles):
            x,y,theta = self.particles1[i].getState()
            particleLocs1[i] = x,y,theta
            
            x,y,theta = self.particles2[i].getState()
            particleLocs2[i] = x,y,theta
    
        return particleLocs1, particleLocs2
    
    
    def plotSubBots(self):
        """
        Plotting where the bots think they are in space
        """
        # First, reset the states of each bot
        self.bot1 = []                       # For triangle of bot 1
        self.bot2 = []                       # For triagle of bot 2
        self.lineAx1.set_data([],[])         # For previous locations of bot 1
        self.lineAx2.set_data([],[])         # For previous locations of bot 2
        self.foundLandmarks1.set_data([],[]) # Landmarks of bot 1 
        self.foundLandmarks2.set_data([],[]) # Landmarks of bot 2
        
        self.getSubStates()

    
    def showState(self):
        """
        Given some plot (or whatever), show where the bot is now
        The bot position is marked using an isoceles triangle
        """
        states = [self.state1, self.state2]
        for i in range(2):
            a =  1                    # This is the length of the longer side of the isoceles traingle
            beta = arctan(sqrt(15)/3) # Calculated analytically. Don't worry about it.
            l = a*sqrt(15)/12         # The height of the triangle divided by 3
            d = sqrt((a**2)/6)        # Length from center of triangle to lower points
            
            x,y,theta = states[i]
            
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
            sensor = plt.Circle([x,y], radius=self.sensorRange)
            self.sensors.append(sensor)
        
    def showSubBot(self, botState, plotIndex):
        """
        Given the calculated average position of the bot, will show the bot 
        as a triangle in the subplot specified by plotIndex.
        """
        a =  1                    # This is the length of the longer side of the isoceles traingle
        beta = arctan(sqrt(15)/3) # Calculated analytically. Don't worry about it.
        l = a*sqrt(15)/12         # The height of the triangle divided by 3
        d = sqrt((a**2)/6)        # Length from center of triangle to lower points
        
        x,y,theta = botState
        
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
        
        # append this polygon to its preoper list
        if plotIndex == 0:
            self.bot1.append(bot)
            self.collectionBot1.set_paths(self.bot1)
        elif plotIndex==1:
            self.bot2.append(bot)
            self.collectionBot2.set_paths(self.bot2)
            
            
            
    def getSubStates(self):
        """
        Draws where the bots THINK they are in the subplots
        """
        botLocs = []            # Where we will store the average location of each bot
        mus = [[],[]]           # Need to average the mean location of each bot
        landmarkMus = [[],[]]   # Storing the suspected location of each landmark
        landmarkLocsX = [[],[]] # The landmark X locations as calculated by average particle Mu
        landmarkLocsY = [[],[]] # The landmark Y locations as calculated by average particle Mu
        lm_indexes = [[],[]]
        lm_seen = [False, False]
        
        initStates = [self.initState1, self.initState2] # Need to convert everything relative to the initial position of the bot
        initStates = np.array(initStates)
        featuresSeen = [self.featuresSeen1, self.featuresSeen2]
        particles = [self.particles1, self.particles2]
        
        for botIndex in range(2):
            if len(featuresSeen[botIndex])>0:
                lm_seen[botIndex]=True
                for feat in featuresSeen[botIndex]:
                    lm_indexes.append(feat)
                    landmarkMus[botIndex].append([])
                
            for particle in particles[botIndex]:
                #Gather particle states
                state = particle.getState()
                mus[botIndex].append(state) 
                
                for i, feat in enumerate(featuresSeen[botIndex]):
                    if lm_seen[botIndex]: # This extra check is probably unnecessary, but we are adding it here just in case.
                        mu, _ = particle.getEKFinfo(feat)
                        landmarkMus[botIndex][i].append(mu)
                
            botLocAllParticles = np.array(mus[botIndex])
            landmarkLocsAllParticles = np.array(landmarkMus[botIndex])
            
            # Calculating transoformation matrix
            _,_,theta = initStates[botIndex]
            T = np.array([[cos(theta),sin(theta)],
                          [-sin(theta),cos(theta)]])
            
            # Calculate the average state based on the average particle headings
            botLoc = np.mean(botLocAllParticles, axis=0)
            botLocCart = botLoc[:2] # Getting the x,y, position
            botLocCart = matmul(T, botLocCart) - initStates[botIndex,:2] # Rotation and translation are occuring here
            botLoc = np.array([botLocCart[0], botLocCart[1], botLoc[2] - theta])
                
            
            self.showSubBot(botLoc, botIndex)
            for j, feat in enumerate(featuresSeen[botIndex]):
                mu = np.mean(landmarkLocsAllParticles[j], axis=0)
                mu = matmul(T,mu) - initStates[botIndex][:2]
                landmarkLocsX[botIndex].append(mu[0])
                landmarkLocsY[botIndex].append(mu[1])
            
            #Now plot the landmarks in their apppropriate subplot
            if botIndex==0:
                self.foundLandmarks1.set_data(landmarkLocsX[botIndex], landmarkLocsY[botIndex])
            elif botIndex==1:
                self.foundLandmarks2.set_data(landmarkLocsX[botIndex], landmarkLocsY[botIndex])
            
                
                                    
    
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
        covs = [[],[]] # Need to average the covariance ellipse 
        mus = [[],[]]  # Need to average the mean location of each landmark
        xs = []   # List where the x-points will be stored
        ys = []   # List where the y-points will be stored
        lm_indexes = [[],[]]
        
        featuresSeen = [self.featuresSeen1, self.featuresSeen2]
        particles = [self.particles1, self.particles2]
        
        for botIndex in range(2):
            if len(featuresSeen[botIndex])>0:
                for feat in featuresSeen[botIndex]:
                    lm_indexes.append(feat)
                    covs[botIndex].append([])
                    mus[botIndex].append([])
                
                for i, feat in enumerate(featuresSeen[botIndex]):
                    for particle in particles[botIndex]:
                        mu, cov = particle.getEKFinfo(feat)
                        mus[botIndex][i].append(mu)
                        covs[botIndex][i].append(cov)
                
                
                mus = np.array(mus)
                covs = np.array(covs)
                
                for j, feat in enumerate(featuresSeen[botIndex]):
                    cov_mat = np.mean(covs[botIndex][j], axis=0)
                    mu = np.mean(mus[botIndex][j], axis=0)
                    
                
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