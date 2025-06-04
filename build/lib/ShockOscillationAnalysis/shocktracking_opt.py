# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 21:59:00 2024

@author: Ahmed H. Hanfy
"""
import numpy as np
import matplotlib.pyplot as plt

def findlocalminimums(Reference, slice_pixels, looping_data, min_point = 0,
                      plot = False, ax = None, threshold = 0, Last_Loc = -1):
   
    strt, end, stp = looping_data
    localmin = [] # .............. Local Minimum set of illumination values
    LocMinI = [] # ......................... Local Minimum set of Locations 
    AeraSet = [] # ............................... local minimum areas list
    for pixel in range(strt,end,stp):
        if slice_pixels[pixel] < Reference: 
        # find the local minimum and store illumination values and location
            localmin.append(slice_pixels[pixel]); LocMinI.append(pixel)
            
            # find open local minimum at the end of the slice
            if pixel == 0 and len(localmin) > 1: 
                A = abs(np.trapz(Reference-localmin))
                SetMinPoint = min(localmin)
                AeraSet.append(A)
                if plot: ax.fill_between(LocMinI, localmin, Reference, alpha=0.5)
                if A > MinA and (Reference-SetMinPoint)/(Reference-min_point) > threshold: 
                    MinA = A;   local_mins_set = [LocMinI, localmin]
                localmin = []; LocMinI = []
        
        # bounded local minimum
        elif slice_pixels[pixel] >= Reference and len(localmin) > 1: 
            A = abs(np.trapz(Reference-localmin))
            SetMinPoint = min(localmin)
            AeraSet.append(A)
            if plot:ax.fill_between(LocMinI, localmin, Reference, alpha=0.5)                        
            if A > MinA and (Reference-SetMinPoint)/(Reference-min_point) > threshold:
                MinA = A;   local_mins_set = [LocMinI,localmin]
            localmin = []; LocMinI = []
            
        else: localmin = [];  LocMinI = []
    return local_mins_set, AeraSet, ax


def ShockTraking(SnapshotSlice, LastShockLoc = -1, 
                 Plot = False, count = -1, Alpha = 0.3):

    # Start processing the slice
    avg = np.mean(SnapshotSlice) # ...... Average illumination on the slice    
    MinimumPoint = min(SnapshotSlice) # ........... minimum (darkest) point 

    if Plot: # to plot slice illumination values with location and Avg. line
        fig, ax = plt.subplots(figsize=(10,5))
        # Plot light intensity
        ax.plot(SnapshotSlice, label = 'Light intensity at certain snapshot')
        # Plot the average line
        ax.axhline(avg, linestyle = ':', color = 'tab:green', label = 'Light intensity average line')
        ax.axhline(MinimumPoint,linestyle = '--',color = 'k')
        ax.set_ylim([-20,255]);  ax.set_yticks(np.arange(0, 260, 51))
        ax.axhline(0,linestyle = ':', color = 'k', alpha = 0.2)
        ax.axhline(255,linestyle = ':', color = 'k', alpha = 0.2)

    # Initiating Variables 
    looping_data = (Pixels-1, -1 , -1)  # ...................... loop-range
    MinA = 0 # ............................................... Minimum Area
    Pixels = len(SnapshotSlice) # ............................. Slice width
    localmin = [] # .............. Local Minimum set of illumination values
    LocMinI = [] # ......................... Local Minimum set of Locations 
    AeraSet = [] # ............................... Local minimum areas list

