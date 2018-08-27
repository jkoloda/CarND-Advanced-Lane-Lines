#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 14:21:23 2018

@author: janko
"""

import numpy as np
from tools import warp_point

# Define a class to receive the characteristics of each line detection

ym_per_pix = 30/float(720) # meters per pixel in y dimension
xm_per_pix = 3.7/600.0 # meters per pixel in x dimension

class Line():
    def __init__(self, niterations, height, width, M):        
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #x values for detected line pixels
        self.centers = None        
        # number of iterations
        self.niterations = niterations
        self.mid_point = warp_point([height, width/2], M)
        self.height = 720
        self.width = 1280
        self.mid_point = warp_point([self.height, self.width/2], M)[1]
        
    def add_recent_xfitted(self, xfitted):
        '''A buffer function. Push one element and throw the oldest one.'''
        self.recent_xfitted.insert(0, xfitted)
        if len(self.recent_xfitted) > self.niterations:
            del self.recent_xfitted[-1]    
    
    def update(self, fit, fitx, fity, centers):
        '''Updates line attributes with current data.'''
        self.add_recent_xfitted(fitx)
        self.bestx = np.mean(np.asarray(self.recent_xfitted), axis=0)
        self.current_fit = fit
        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            # Moving average
            self.best_fit = (self.current_fit + self.best_fit)/2.0
        
        # Fit new polynomials to x,y in world space
        self.centers = centers
        fit_cr = np.polyfit(self.centers[:,0]*ym_per_pix, self.centers[:,1]*xm_per_pix, 2)
        # Calculate the new radii of curvature
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*self.height*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
                
        self.line_base_pos = xm_per_pix * (self.mid_point - self.bestx[-1])
        